# LLM From Scratch

A custom language model project built from scratch for training, fine-tuning, evaluation, and interactive chat.  
This repository includes scripts for pretraining, supervised fine-tuning, evaluation, a simple chat server, tokenizer assets, checkpoints, and a small UI.

## Features

- Train a GPT-style language model from scratch
- Fine-tune the base model on instruction/chat data
- Evaluate model quality on test prompts
- Run local chat inference through a Python script
- Launch a chat server for interactive use
- Organize experiments with multiple checkpoint versions
- Includes tokenizer and data preparation assets

## Project Structure

├── checkpoints/          
├── checkpoints_sft/      
├── checkpoints_v2/       
├── checkpoints_v3/       
├── data/                 
├── scripts/              
├── tokenizer/             
├── ui/                   
├── chat_sample.py        
├── chat_server.py        
├── eval_test.py          
├── sample.py             
├── sft_train.py          
├── train_gpt.py          
├── requirements.txt      
└── README.md             

## Installation

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Training

### Pretraining
python train_gpt.py

### Supervised Fine-Tuning
python sft_train.py

## Inference

python sample.py
python chat_sample.py

## Chat Server

python chat_server.py

## Evaluation

python eval_test.py

## Notes

- Model quality depends on dataset and training setup
- Large models require strong compute resources

## License

Add your license here.
