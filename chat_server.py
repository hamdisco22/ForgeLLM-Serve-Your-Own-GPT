from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
import sentencepiece as spm
import string
from train_gpt import GPT

PROMPT_TEMPLATE = "### Instruction:\n{inst}\n\n### Response:\n"

# -------------------------
# FastAPI app + CORS
# -------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatReq(BaseModel):
    message: str
    max_new_tokens: int = 140
    # Force stable decoding by default
    temperature: float = 0.2
    top_k: int = 1  # 1 => greedy
    repetition_penalty: float = 1.15


# -------------------------
# Load model once
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "checkpoints_sft/best.pt"
SPM_PATH = "tokenizer/spm.model"

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
cfg = ckpt["config"]
meta = ckpt["meta"]

sp = spm.SentencePieceProcessor()
sp.load(SPM_PATH)

eos_id = sp.eos_id() if sp.eos_id() != -1 else None
bos_id = sp.bos_id()

model = GPT(
    vocab_size=int(meta["vocab_size"]),
    block_size=int(cfg["block_size"]),
    n_layer=int(cfg["n_layer"]),
    n_head=int(cfg["n_head"]),
    n_embd=int(cfg["n_embd"]),
    dropout=float(cfg.get("dropout", 0.1)),
).to(DEVICE)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

# -------------------------
# Token guards to reduce junk
# -------------------------
PUNCT_CHARS = set(string.punctuation) | {"’", "“", "”", "…", "—", "–", "«", "»"}
def _clean_piece(piece: str) -> str:
    # SentencePiece may include ▁ (whitespace marker)
    return piece.replace("▁", "").strip()

def build_bad_id_sets(sp: spm.SentencePieceProcessor):
    bad_start_ids = set()
    punct_ids = set()

    vocab = sp.get_piece_size()
    for i in range(vocab):
        p = _clean_piece(sp.id_to_piece(i))
        if not p:
            # empty/space-like: bad start and punct-like
            bad_start_ids.add(i)
            punct_ids.add(i)
            continue

        # purely punctuation => punct id
        if all(ch in PUNCT_CHARS for ch in p):
            punct_ids.add(i)

        # disallow starting with heavy punctuation / separators
        if p[0] in PUNCT_CHARS:
            bad_start_ids.add(i)

        # also block obvious template header tokens at start
        if p in {"###", "Instruction", "Response", ":", "\n"}:
            bad_start_ids.add(i)

    return bad_start_ids, punct_ids

BAD_START_IDS, PUNCT_IDS = build_bad_id_sets(sp)


@torch.no_grad()
def generate(
    model,
    x,
    max_new_tokens: int,
    eos_id=None,
    temperature: float = 0.2,
    top_k: int = 1,
    repetition_penalty: float = 1.15,
):
    model.eval()
    greedy = (top_k == 1) or (temperature <= 0.2)

    same_token_run = 0
    last_token = None

    for t in range(max_new_tokens):
        x_cond = x[:, -model.block_size:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :]

        # penalize repeating already-used tokens
        if repetition_penalty and repetition_penalty > 1.0:
            used = set(x[0].tolist())
            logits[0, list(used)] /= repetition_penalty

        # block bad starts for the first generated token
        if t == 0 and BAD_START_IDS:
            logits[0, list(BAD_START_IDS)] = -float("inf")

        # temperature
        temperature = max(1e-6, float(temperature))
        logits = logits / temperature

        if greedy:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # top-k sampling
            if top_k is not None and int(top_k) > 0:
                k = min(int(top_k), logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        nid = int(next_id.item())
        x = torch.cat([x, next_id], dim=1)

        # stop on EOS
        if eos_id is not None and nid == int(eos_id):
            break

        # stop if model loops the exact same token too long
        if last_token == nid:
            same_token_run += 1
        else:
            same_token_run = 0
        last_token = nid
        if same_token_run >= 25:
            break

        # stop if last 40 tokens are mostly punctuation
        tail = x[0].tolist()[-40:]
        if len(tail) >= 40:
            punct_ratio = sum(1 for z in tail if z in PUNCT_IDS) / 40.0
            if punct_ratio >= 0.85:
                break

    return x


@app.get("/")
def root():
    return {"status": "ok", "device": DEVICE, "ckpt": CKPT_PATH}


@app.post("/chat")
def chat(req: ChatReq):
    inst = (req.message or "").strip()
    if not inst:
        return {"answer": "Please type a message."}

    prompt = PROMPT_TEMPLATE.format(inst=inst)
    prompt_ids = sp.encode(prompt, out_type=int)
    if bos_id != -1:
        prompt_ids = [bos_id] + prompt_ids

    x = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    y = generate(
        model,
        x,
        max_new_tokens=int(req.max_new_tokens),
        eos_id=eos_id,
        temperature=float(req.temperature),
        top_k=int(req.top_k),
        repetition_penalty=float(req.repetition_penalty),
    )

    out_ids = y[0].tolist()
    new_ids = out_ids[len(prompt_ids):]

    if eos_id is not None and eos_id in new_ids:
        new_ids = new_ids[: new_ids.index(eos_id)]

    answer = sp.decode(new_ids).strip()

    # final cleanup: remove accidental headers
    if "###" in answer:
        answer = answer.split("###", 1)[0].strip()
    answer = answer.lstrip(" \n\r\t:=-,").strip()

    if not answer:
        answer = "(empty response)"

    return {"answer": answer}
