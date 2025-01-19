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
TRAIN_DATASET_PATH = "./datasets/train.json"
VAL_DATASET_PATH = "./datasets/val.json"
TEST_DATASET_PATH = "./datasets/test.json"
LORA_DIR = "./local/lora_weights"  # New constant for LoRA directory
LORA_SAVE_NAME = f"{LORA_DIR}/70b-4bit-lora-r128"  # Updated path
LORA_DIM = 128
BATCH_SIZE = 2
GRAD_ACCUM = 4

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

train_dataset = Dataset.from_list([{"text": formatting_prompts_func(d)} for d in load_json_data(TRAIN_DATASET_PATH)])
val_dataset = Dataset.from_list([{"text": formatting_prompts_func(d)} for d in load_json_data(VAL_DATASET_PATH)])

def parse_args():
    parser = argparse.ArgumentParser(description='Train a language model using unsloth')
    parser.add_argument('--load_in_4bit', type=bool, default=True,
                        help='Use 4bit quantization (default: True)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs (default: 1)')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from (default: None)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set wandb directory
    os.environ["WANDB_DIR"] = "./local/wandb"
    os.makedirs("./local/wandb", exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="narrative-finetune",
        run_name=f"lora-r{LORA_DIM}-bs{BATCH_SIZE}",
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
    os.makedirs(LORA_DIR, exist_ok=True)
    
    # Set variables from command line arguments and constants
    max_seq_length = MAX_SEQ_LENGTH
    dtype = torch.bfloat16  # Keeping this fixed as it's hardware dependent
    load_in_4bit = args.load_in_4bit
    cache_dir = CACHE_DIR
    model_name = MODEL_NAME
    train_dataset_path = TRAIN_DATASET_PATH
    val_dataset_path = VAL_DATASET_PATH
    test_dataset_path = TEST_DATASET_PATH
    lora_save_name = LORA_SAVE_NAME
    lora_dim = LORA_DIM

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
            output_dir = "outputs",
            logging_dir = "logs",
            report_to = "wandb",
            eval_strategy = "steps",
            eval_steps = 10,
            save_strategy = "steps",
            save_steps = 10,
            save_total_limit = 20,
        ),
    )

    trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.save_pretrained(lora_save_name)
    tokenizer.save_pretrained(lora_save_name)
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()

