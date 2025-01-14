#!/usr/bin/env python
# finetune_lora.py

import os
import json
import random
from typing import Dict, List, Any

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    TrainingArguments,
    Trainer,
)

# ---------------------------
# Unsloth-specific imports
# ---------------------------
from unsloth import FastLanguageModel, is_bfloat16_supported

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
    synopsis_text = " ".join(example["synopsis"])
    instruction_text = (
        """Identify the turning points and their sentence number in this narrative."
        " Then provide the final turning points' location clearly in a json format, like {"tp1": ##.#, "tp2": ##.#, "tp3": ##.#, "tp4": ##.#, "tp5": ##.#}.

        ### TURNING POINT DEFINITIONS
        1. **Opportunity** – Introductory event that occurs after presenting the setting and background of the main characters.
        2. **Change of Plans** – Event where the main goal of the story is defined, starting the main action.
        3. **Point of No Return** – Event that pushes the main character(s) to fully commit to their goal.
        4. **Major Setback** – Event where things fall apart temporarily or permanently.
        5. **Climax** – Final event/resolution of the main story (the “biggest spoiler”).
        """
    )

    turning_points_str = json.dumps(example["turning_points"])
    output_text = (
        f"{example['tp_pred_reasoning']}\n\n"
        f"Final turning points: {turning_points_str}"
    )

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
    """
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
        tokenized["input_ids"][-1] = tokenizer.eos_token_id
    return tokenized


def main():
    # --------------------------
    # Hyperparameters / Settings
    # --------------------------
    train_json_path = "train.json"
    val_json_path = "val.json"
    model_name = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"  # Example Unsloth model
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

    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data)
    })

    # -------------
    # Tokenizer
    # -------------
    # Note: FastLanguageModel.from_pretrained(...) will provide you
    # with both the model and the tokenizer.
    def tokenize_wrap(example):
        return tokenize_function(example, tokenizer, max_length=cutoff_len)

    # ------------------------------------------------
    # Load Model & Tokenizer from Unsloth in 4-bit
    # ------------------------------------------------
    # This automatically handles quantization + GPU mapping for you.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=cutoff_len,  # You can adjust this as needed
        dtype=None,                 # Let Unsloth handle dtypes
        load_in_4bit=True,          # 4-bit quantized
        device_map="auto",          # Auto-distribute across available GPUs
        cache_dir=cache_dir,
    )

    # ---------------------------------------------------
    # Convert the model to a LoRA model (PEFT) via Unsloth
    # ---------------------------------------------------
    # * For LLaMA-based models, typical target modules include q_proj, k_proj, v_proj, etc.
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
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

    # ------------------------
    # Tokenize the HF Dataset
    # ------------------------
    tokenized_dataset = dataset_dict.map(tokenize_wrap, batched=False)

    # Simple data collator
    def data_collator(features):
        batch = {k: [f[k] for f in features] for k in features[0].keys()}
        batch = tokenizer.pad(
            batch,
            padding=True,
            return_tensors="pt"
        )
        return batch

    # -----------------------------------------
    # TrainingArguments & Trainer Initialization
    # -----------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        eval_steps=50,
        logging_steps=50,
        load_best_model_at_end=False,
        report_to="none",
        # Enable fp16 unless the environment supports bf16
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        # 8-bit AdamW often helps with 4-bit models
        optim="adamw_8bit"
    )

    # Initialize the HF Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    # ----------------
    # Start Fine-Tuning
    # ----------------
    print("Starting LoRA fine-tuning with Unsloth...")
    trainer.train()

    # ------------
    # Save Model
    # ------------
    print(f"Saving LoRA adapters (and config) to {output_dir}...")
    trainer.save_model(output_dir)

    # Optionally merge LoRA weights into the base model:
    merged_model = FastLanguageModel.merge_and_unload(model)
    merged_model.save_pretrained(output_dir)

    print("Done!")


if __name__ == "__main__":
    main()