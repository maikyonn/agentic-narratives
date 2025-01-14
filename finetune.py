#!/usr/bin/env python
# finetune_lora.py

import os
import json
import random
import torch
from typing import Dict, List, Any
from datasets import Dataset, DatasetDict

# Hugging Face Transformers and PEFT (LoRA) imports
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, PeftType, TaskType


# ------------------------------------------------------------------------------
# 1. Load and preprocess data
# ------------------------------------------------------------------------------
def load_json_data(json_path: str) -> List[Dict[str, Any]]:
    """
    Load your narrative data from a JSON file.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def format_example(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert a single data point into an instruction-style prompt/response.
    You can customize this to match your desired format.
    """
    # For clarity, we’ll just join the synopsis into one text block.
    synopsis_text = " ".join(example["synopsis"])

    # The instruction:
    instruction_text = (
        """Identify the turning points and their sentence number in this narrative and explain your reasoning step-by-step."
        " Then provide the final turning points' location clearly in a json format, like {"tp1": ##.#, "tp2": ##.#, "tp3": ##.#, "tp4": ##.#, "tp5": ##.#}.

        ### TURNING POINT DEFINITIONS
        1. **Opportunity** – Introductory event that occurs after presenting the setting and background of the main characters.
        2. **Change of Plans** – Event where the main goal of the story is defined, starting the main action.
        3. **Point of No Return** – Event that pushes the main character(s) to fully commit to their goal.
        4. **Major Setback** – Event where things fall apart temporarily or permanently.
        5. **Climax** – Final event/resolution of the main story (the “biggest spoiler”).

        """

        
    )

    # We’ll place the chain-of-thought inside the response, along with the final turning points.
    # You already have 'turning_points' and 'tp_pred_reasoning' in your data.
    # We'll combine them into a single output string.
    turning_points_str = json.dumps(example["turning_points"])

    # Here we combine the chain-of-thought and the final turning points
    # in a single output for training. Adjust as you see fit.
    output_text = (
        f"{example['tp_pred_reasoning']}\n\n"
        f"Final turning points: {turning_points_str}"
    )

    # Return an “instruction” style dictionary
    return {
        "instruction": instruction_text,
        "input": synopsis_text,
        "output": output_text
    }


# ------------------------------------------------------------------------------
# 2. Tokenization & Dataset Preparation
# ------------------------------------------------------------------------------
def tokenize_function(
    example: Dict[str, str],
    tokenizer,
    max_length: int = 512,
    add_eos_token: bool = True,
) -> Dict[str, List[int]]:
    """
    Tokenize the instruction + input + output into a single text sequence.
    We place the output after a special separator or prompt design.
    """
    # Create a prompt that includes the instruction and the input
    # You can adjust this format to your preference.
    prompt_text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )
    tokenized = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    if add_eos_token and tokenized["input_ids"][-1] != tokenizer.eos_token_id:
        # Replace the last token with eos if it doesn’t already exist
        tokenized["input_ids"][-1] = tokenizer.eos_token_id
    return tokenized


# ------------------------------------------------------------------------------
# 3. Main Fine-Tuning Logic
# ------------------------------------------------------------------------------
def main():
    # --------------------------
    # Hyperparameters / Settings
    # --------------------------
    train_json_path = "train.json"
    val_json_path = "val.json"
    model_name = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
    output_dir = "./lora-finetuned-llama"
    cache_dir = "./model_cache"
    max_steps = 1000
    batch_size = 2
    gradient_accumulation_steps = 8
    learning_rate = 1e-4
    cutoff_len = 512

    # -------------
    # Load the Data
    # -------------
    train_data = [format_example(d) for d in load_json_data(train_json_path)]
    val_data = [format_example(d) for d in load_json_data(val_json_path)]

    # Create Hugging Face Datasets
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data)
    })

    # -------------
    # Tokenizer
    # -------------
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    # If using Llama-based models, they sometimes require special handling (e.g., setting pad_token)
    # For LLaMA-2, you often need to do:
    # tokenizer.pad_token = tokenizer.eos_token
    # Or if it’s a chat model, you might adapt the approach.

    def tokenize_wrap(example):
        return tokenize_function(example, tokenizer, max_length=cutoff_len)

    tokenized_dataset = dataset_dict.map(tokenize_wrap, batched=False)

    # Data collator for causal language modeling
    def data_collator(features):
        # If you're using the default DataCollatorForLanguageModeling
        # from transformers, it might be okay, but for custom generation tasks,
        # you can write your own here. For simplicity, let's do basic padding:
        batch = {k: [f[k] for f in features] for k in features[0].keys()}
        batch = tokenizer.pad(
            batch,
            padding=True,
            return_tensors="pt"
        )
        return batch

    # -----------------------------
    # Load Base Model + Apply LoRA
    # -----------------------------
    print(torch.cuda.memory_allocated())
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # or "auto"
        device_map="cuda:0",           # automatically place on GPU
        cache_dir=cache_dir
    )
    print(torch.cuda.memory_allocated())

    # Set up LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Typically used for LLaMA-based models
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM  # We’re doing a causal language modeling task
    )
    print(torch.cuda.memory_allocated())

    # Add LoRA adapters to the base model
    lora_model = get_peft_model(base_model, peft_config)

    # -----------------------------------------
    # TrainingArguments & Trainer Initialization
    # -----------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # We'll just set 1 epoch and rely on max_steps if we want
        max_steps=max_steps,  # Usually used for larger datasets
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=100,        # Adjust as needed
        eval_steps=50,         # Adjust as needed
        logging_steps=50,      # Adjust as needed
        load_best_model_at_end=False,  # Set True if you want to restore best model
        report_to="none",      # or "wandb" if you want to track in Weights & Biases
        fp16=True,
        optim="adamw_torch"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )
    print(torch.cuda.memory_allocated())

    # ----------------
    # Start Fine-Tuning
    # ----------------
    print("Starting LoRA fine-tuning...")
    trainer.train()

    # ------------
    # Save Model
    # ------------
    print(f"Saving LoRA adapters to {output_dir}...")
    trainer.save_model(output_dir)

    # If you want to merge LoRA weights with the base model,
    # you can do so after training (optional). For example:
    # lora_model = lora_model.merge_and_unload()
    # lora_model.save_pretrained(output_dir)

    print("Done!")



if __name__ == "__main__":
    main()
