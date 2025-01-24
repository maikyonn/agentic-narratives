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
)
from transformers.utils import logging
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import wandb  # Add wandb import

# Constants
MAX_SEQ_LENGTH = 8192
CACHE_DIR = "./local/model_cache"
MODEL_NAME = "unsloth/Llama-3.3-70B-Instruct"
TEST_DATASET_PATH = "./datasets/test.json"
BATCH_SIZE = 2
GRAD_ACCUM = 4
LORA_DIM = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = torch.bfloat16, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    cache_dir=CACHE_DIR,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_DIM, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

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

def formatting_prompts_func(examples: List[Dict[str, Any]]) -> List[str]:
    """
    Format prompts by combining instruction, input, and output from each example.
    """
    output_texts = []
    for example in examples:
        text = (
            f"{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n" 
            f"### Response:\n{example['output']}"
        ) + tokenizer.eos_token
        output_texts.append(text)
    return output_texts

def parse_args():
    parser = argparse.ArgumentParser(description='Train a language model using unsloth')
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to training dataset JSON file')
    parser.add_argument('--val_path', type=str, required=True,
                        help='Path to validation dataset JSON file')
    parser.add_argument('--load_in_4bit', type=bool, default=True,
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
    
    # Create LoRA weights directory if it doesn't exist
    os.makedirs(lora_dir, exist_ok=True)
    
    # Load and format training data
    train_data = load_json_data(args.train_path)
    formatted_train_texts = formatting_prompts_func(train_data)
    train_dataset = Dataset.from_list([{"text": text} for text in formatted_train_texts])

    # Load and format validation data  
    val_data = load_json_data(args.val_path)
    formatted_val_texts = formatting_prompts_func(val_data)
    val_dataset = Dataset.from_list([{"text": text} for text in formatted_val_texts])

    # Set variables from command line arguments and constants
    max_seq_length = MAX_SEQ_LENGTH
    dtype = torch.bfloat16  # Keeping this fixed as it's hardware dependent
    load_in_4bit = args.load_in_4bit
    cache_dir = CACHE_DIR
    model_name = MODEL_NAME
    test_dataset_path = TEST_DATASET_PATH

    # Update training arguments with command line params
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            warmup_steps = 5,
            num_train_epochs = args.epochs,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
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
        ),
    )

    trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save locally first
    model.save_pretrained(lora_save_name)
    tokenizer.save_pretrained(lora_save_name)
    
    # Push to Hub
    print(f"Uploading model to HuggingFace Hub: {args.hub_model_id}")
    model.push_to_hub(args.hub_model_id, use_temp_dir=True)
    tokenizer.push_to_hub(args.hub_model_id, use_temp_dir=True)
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
