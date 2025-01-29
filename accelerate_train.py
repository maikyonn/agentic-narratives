import os
import json
import random
import argparse
from typing import Dict, List, Any

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer
)
from transformers.utils import logging
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed

# Constants
MAX_SEQ_LENGTH = 8192
CACHE_DIR = "./local/model_cache"
MODEL_NAME = "unsloth/Llama-3.3-70B-Instruct"
TEST_DATASET_PATH = "./datasets/test.json"
BATCH_SIZE = 2
GRAD_ACCUM = 4
LORA_DIM = 128

# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    load_in_4bit=False,
    device_map="auto",
    cache_dir=CACHE_DIR
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    padding_side="right",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
lora_config = LoraConfig(
    r=LORA_DIM,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM"
)

# Get PEFT model
model = get_peft_model(model, lora_config)
model.enable_gradient_checkpointing()
def formatting_prompts_func(examples: List[Dict[str, Any]]) -> List[str]:
    """
    Format prompts by appending EOS token to complete-text.
    """
    output_texts = []
    for example in examples:
        text = example['complete-text'] + tokenizer.eos_token
        output_texts.append(text)
    return output_texts

def load_json_data(json_path: str) -> List[Dict[str, Any]]:
    """
    Load narrative data from a JSON file.
    This function wraps multiple JSON objects into a list if they aren't already.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            data = []
    
    return data



def parse_args():
    parser = argparse.ArgumentParser(description='Train a language model using accelerate')
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to training dataset JSON file')
    parser.add_argument('--val_path', type=str, required=True,
                        help='Path to validation dataset JSON file')
    parser.add_argument('--load_in_4bit', type=bool, default=False,
                        help='Use 4bit quantization (default: True)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs (default: 1)')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from (default: None)')
    parser.add_argument('--hub_model_id', type=str, 
                        default="owaridere/Llama-3.3-70B-Instruct-tp-finetune",
                        help='Model ID for uploading to HuggingFace Hub')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Set seed for reproducibility
    set_seed(3407)
    
    # Move existing wandb init inside main process only
    if accelerator.is_main_process:
        # Set lora_dir to be in the same directory as train_path
        lora_dir = os.path.dirname(args.train_path)
        lora_save_name = os.path.join(lora_dir, "70b-4bit-lora-r128")
        
        # Set wandb directory to be in lora_dir
        os.environ["WANDB_DIR"] = os.path.join(lora_dir, "wandb")
        os.makedirs(os.path.join(lora_dir, "wandb"), exist_ok=True)
        
        # Initialize wandb
        wandb.init(
            project="narrative-finetune",
            name=f"lora-r{LORA_DIM}-bs{BATCH_SIZE}",
            config={
                "model_name": MODEL_NAME,
                "lora_dim": LORA_DIM,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM,
                "max_seq_length": MAX_SEQ_LENGTH,
                "epochs": args.epochs,
                "learning_rate": 2e-4,
            }
        )

    train_data = load_json_data(args.train_path)
    formatted_train_texts = formatting_prompts_func(train_data)
    train_dataset = Dataset.from_list([{"text": text} for text in formatted_train_texts])

    # Load and format validation data  
    val_data = load_json_data(args.val_path)
    formatted_val_texts = formatting_prompts_func(val_data)
    val_dataset = Dataset.from_list([{"text": text} for text in formatted_val_texts])


    # Update training arguments for distributed training
    training_args = TrainingArguments(
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        warmup_steps = 5,
        num_train_epochs = args.epochs,
        learning_rate = 2e-4,
        fp16 = False,
        bf16 = True,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = os.path.join(lora_dir, "outputs"),
        logging_dir = os.path.join(lora_dir, "logs"),
        report_to = "wandb",
        eval_strategy = "steps",
        eval_steps = 5,
        save_strategy = "steps",
        save_steps = 10,
        save_total_limit = 20,
        # Add these for distributed training
        ddp_find_unused_parameters = False,
        gradient_checkpointing = True,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False,
        args = training_args,
    )

    trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    print(trainer_stats)
    # Save only on main process
    if accelerator.is_main_process:
        model.save_pretrained(lora_save_name)
        tokenizer.save_pretrained(lora_save_name)
        
        # Close wandb run
        wandb.finish()

if __name__ == "__main__":
    main()
