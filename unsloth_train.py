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

max_seq_length = 8192 # Choose any! We auto support RoPE Scaling internally!
dtype = torch.bfloat16 # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
cache_dir = "./model_cache-2"
model_name = "unsloth/Llama-3.3-70B-Instruct"
train_dataset_path = "./datasets/train.json"
val_dataset_path = "./datasets/val.json"
test_dataset_path = "./datasets/test.json"
lora_save_name = "70b-4bit-lora-r128"
lora_dim = 128

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    cache_dir=cache_dir,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_dim, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
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

import json
from pprint import pprint

def load_json_data(json_path: str) -> List[Dict[str, Any]]:
    """
    Load your narrative data from a JSON file.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Synopsis:
{}

### Response:
{}

Final Turning Points: {}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instruction = (
    """Identify the turning points and their sentence number in this narrative and explain your reasoning step-by-step. Then provide the final turning point locations clearly in a json format, like {"tp1": ##.#, "tp2": ##.#, "tp3": ##.#, "tp4": ##.#, "tp5": ##.#}.

    ### TURNING POINT DEFINITIONS
    1. **Opportunity** – Introductory event that occurs after presenting the setting and background of the main characters.
    2. **Change of Plans** – Event where the main goal of the story is defined, starting the main action.
    3. **Point of No Return** – Event that pushes the main character(s) to fully commit to their goal.
    4. **Major Setback** – Event where things fall apart temporarily or permanently.
    5. **Climax** – Final event/resolution of the main story (the "biggest spoiler").
    """ 
)
    input       = "\n".join(f"{i+1}. {sent}" for i, sent in enumerate(examples["synopsis"]))
    output      = examples["tp_pred_reasoning"]
    final_tp    = examples["turning_points"]
    texts = []
    return alpaca_prompt.format(instruction, input, output, final_tp) + EOS_TOKEN

pass

train_dataset = Dataset.from_list([{"text": formatting_prompts_func(d)} for d in load_json_data(train_dataset_path)])
val_dataset = Dataset.from_list([{"text": formatting_prompts_func(d)} for d in load_json_data(val_dataset_path)])

def parse_args():
    parser = argparse.ArgumentParser(description='Train a language model using unsloth')
    parser.add_argument('--max_seq_length', type=int, default=8192,
                        help='Maximum sequence length (default: 8192)')
    parser.add_argument('--load_in_4bit', type=bool, default=True,
                        help='Use 4bit quantization (default: True)')
    parser.add_argument('--cache_dir', type=str, default='./model_cache-2',
                        help='Cache directory for models')
    parser.add_argument('--model_name', type=str, default='unsloth/Llama-3.3-70B-Instruct',
                        help='Name of the model to use')
    parser.add_argument('--train_dataset', type=str, default='./datasets/train.json',
                        help='Path to training dataset')
    parser.add_argument('--val_dataset', type=str, default='./datasets/val.json',
                        help='Path to validation dataset')
    parser.add_argument('--test_dataset', type=str, default='./datasets/test.json',
                        help='Path to test dataset')
    parser.add_argument('--lora_save_name', type=str, default='70b-4bit-lora-r128',
                        help='Name for saving the LoRA weights')
    parser.add_argument('--lora_dim', type=int, default=128,
                        help='LoRA dimension (default: 128)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Per device batch size (default: 2)')
    parser.add_argument('--grad_accum', type=int, default=4,
                        help='Gradient accumulation steps (default: 4)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs (default: 1)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set variables from command line arguments
    max_seq_length = args.max_seq_length
    dtype = torch.bfloat16  # Keeping this fixed as it's hardware dependent
    load_in_4bit = args.load_in_4bit
    cache_dir = args.cache_dir
    model_name = args.model_name
    train_dataset_path = args.train_dataset
    val_dataset_path = args.val_dataset
    test_dataset_path = args.test_dataset
    lora_save_name = args.lora_save_name
    lora_dim = args.lora_dim

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
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = args.grad_accum,
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
            output_dir = "outputs",
            logging_dir = "logs",
            report_to = "none",
            eval_strategy = "steps",
            eval_steps = 10,
        ),
    )

    trainer_stats = trainer.train(resume_from_checkpoint = "outputs/checkpoint-41")
    model.save_pretrained(lora_save_name)
    tokenizer.save_pretrained(lora_save_name)

if __name__ == "__main__":
    main()

