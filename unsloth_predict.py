#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage:
    python generate_val_predictions.py [--use_lora]

This script:
  - Loads val.json (list of data entries).
  - Loads either the base model or the LoRA model (depending on --use_lora).
  - For each entry in val.json, if the CSV row for that entry is incomplete,
    it performs inference, updates that row, and immediately writes
    the updated data to 'val_predictions.csv'.

Columns in val_predictions.csv:
  - index
  - original_text
  - non_lora_response
  - lora_response
  - ground_truth
"""

import os
import re
import gc
import csv
import json
import torch
import argparse
from typing import Dict, List, Any, Optional
from transformers import TextStreamer
from datasets import Dataset
from unsloth import FastLanguageModel

#######################################
# 1) Parse command-line arguments
#######################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If provided, load the LoRA model instead of the base model."
    )
    return parser.parse_args()


#######################################
# 2) Helper functions
#######################################
def load_json_data(json_path: str) -> List[Dict[str, Any]]:
    """
    Load narrative data from a JSON file.
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

def formatting_prompts_func(example: Dict[str, Any], EOS_TOKEN: str) -> str:
    """
    Build the text prompt from a single example. 
    """
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
    input_part = "\n".join(f"{i+1}. {sent}" for i, sent in enumerate(example["synopsis"]))
    output_part = example.get("tp_pred_reasoning", "")
    final_tp = example.get("turning_points", "")

    return alpaca_prompt.format(
        instruction,
        input_part,
        output_part,
        final_tp
    ) + EOS_TOKEN

def load_model(use_lora: bool = False):
    """
    Load the model and tokenizer. If 'use_lora' is True, load the LoRA adapter. 
    Otherwise, load the base model. 
    """
    max_seq_length = 8192   # We auto support RoPE Scaling internally in unsloth
    dtype = torch.bfloat16  # Use bfloat16 if GPU is Ampere+
    load_in_4bit = True     # Use 4-bit quantization to reduce memory usage
    cache_dir = "./model_cache-2"

    if use_lora:
        # Load the LoRA adapter weights (replace "lora_model" as needed)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="70b-4bit-lora-r128-epoch1",
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            cache_dir=cache_dir
        )
    else:
        # Load the base model (replace with your actual base model)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Llama-3.3-70B-Instruct",
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            # token="hf_...",  # If you have a gated model
            cache_dir=cache_dir,
        )

    FastLanguageModel.for_inference(model)

    # We'll build the val dataset externally or on-the-fly. 
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str) -> str:
    """
    Generate a response from the model given a prompt (omitting the '### Response:' part).
    """
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    # We can omit streaming or set your own parameters
    generated_tokens = model.generate(
        **inputs,
        max_new_tokens=8192,
        # Additional generation params, e.g. temperature, top_p, etc.
    )
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

def read_existing_predictions(predictions_path: str) -> List[Dict[str, Any]]:
    """
    Read existing predictions JSON file if it exists.
    If not, return None.
    """
    if os.path.exists(predictions_path):
        with open(predictions_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def write_predictions(predictions_path: str, data: List[Dict[str, Any]]):
    """
    Write predictions to JSON file.
    """
    with open(predictions_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

#######################################
# 3) Main
#######################################
def main():
    args = parse_args()
    use_lora = args.use_lora

    # 1) Load val data
    val_data = load_json_data("datasets/val.json")
    
    # 2) Load or create predictions file
    predictions_path = "val_predictions.json"
    existing_predictions = read_existing_predictions(predictions_path)
    
    if existing_predictions is None:
        # Create a deep copy of val_data and add new fields
        predictions = []
        for item in val_data:
            pred_item = item.copy()
            pred_item['non_lora_response'] = ""
            pred_item['lora_response'] = ""
            predictions.append(pred_item)
    else:
        predictions = existing_predictions

    # 3) Load model (base or LoRA)
    model, tokenizer = load_model(use_lora=use_lora)
    EOS_TOKEN = tokenizer.eos_token

    # 4) Iterate over each example
    for i, item in enumerate(predictions):
        # Based on whether we are using LoRA or not, check if we should generate
        if not use_lora:
            # We only fill the non-LoRA response if it's empty
            if not item.get('non_lora_response'):
                # Format the prompt
                prompt_text = formatting_prompts_func(item, EOS_TOKEN)
                split_parts = prompt_text.split("### Response:\n")
                prompt_for_inference = split_parts[0]

                # Generate
                response = generate_response(model, tokenizer, prompt_for_inference)
                item['non_lora_response'] = response

                # Immediately write JSON
                write_predictions(predictions_path, predictions)
                print(f"[Non-LoRA] Processed index {i}, wrote to JSON.")

        else:
            # Using LoRA
            if not item.get('lora_response'):
                prompt_text = formatting_prompts_func(item, EOS_TOKEN)
                split_parts = prompt_text.split("### Response:\n")
                prompt_for_inference = split_parts[0]

                # Generate
                response = generate_response(model, tokenizer, prompt_for_inference)
                item['lora_response'] = response

                # Immediately write JSON
                write_predictions(predictions_path, predictions)
                print(f"[LoRA] Processed index {i}, wrote to JSON.")

    # 5) Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    print("Done! If you re-run this script, it will skip entries already filled.")


if __name__ == "__main__":
    main()
