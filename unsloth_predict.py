#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage:
    python generate_val_predictions.py [--lora_name]

This script:
  - Loads val.json (list of data entries).
  - Loads either the base model or the LoRA model (depending on --lora_name).
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
        "--lora_name",
        type=str,
        default=None,
        help="Path to the LoRA model directory. If not provided, uses base model."
    )
    parser.add_argument(
        "--val_path",
        type=str,
        required=True,
        help="Path to validation dataset JSON file"
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
    Format prompts by combining instruction and input, leaving response blank for generation.
    """
    text = (
        f"{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n" 
        f"### Response:\n"
    ) + EOS_TOKEN
    return text

def load_model(lora_name: Optional[str] = None):
    """
    Load the model and tokenizer. If 'lora_name' is provided, load the LoRA adapter. 
    Otherwise, load the base model. 
    """
    max_seq_length = 8192   # We auto support RoPE Scaling internally in unsloth
    dtype = torch.bfloat16  # Use bfloat16 if GPU is Ampere+
    load_in_4bit = True     # Use 4-bit quantization to reduce memory usage
    cache_dir = "./model_cache-2"

    if lora_name:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            cache_dir=cache_dir
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Llama-3.3-70B-Instruct",
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            cache_dir=cache_dir,
        )

    FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str) -> str:
    """
    Generate a response from the model given a prompt.
    """
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    generated_tokens = model.generate(
        **inputs,
        max_new_tokens=8192,
    )
    response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    
    # Extract only the response part after "### Response:"
    if "### Response:" in response:
        response = response.split("### Response:")[1].strip()
    return response

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

    # 1) Load val data
    val_data = load_json_data(args.val_path)
    
    # 2) Set predictions path to same directory as val_path
    val_dir = os.path.dirname(args.val_path)
    predictions_path = os.path.join(val_dir, "val_predictions.json")
    existing_predictions = read_existing_predictions(predictions_path)
    
    if existing_predictions is None:
        predictions = []
        for item in val_data:
            pred_item = item.copy()
            pred_item['non_lora_response'] = ""
            pred_item['lora_response'] = ""
            predictions.append(pred_item)
    else:
        predictions = existing_predictions

    # 3) Load model (base or LoRA)
    model, tokenizer = load_model(lora_name=args.lora_name)
    EOS_TOKEN = tokenizer.eos_token

    # 4) Iterate over each example
    for i, item in enumerate(predictions):
        response_key = 'lora_response' if args.lora_name else 'non_lora_response'
        
        if not item.get(response_key):
            prompt_text = formatting_prompts_func(item, EOS_TOKEN)
            response = generate_response(model, tokenizer, prompt_text)
            item[response_key] = response

            # Immediately write JSON
            write_predictions(predictions_path, predictions)
            print(f"[{'LoRA' if args.lora_name else 'Non-LoRA'}] Processed index {i}, wrote to JSON.")

    # 5) Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    print("Done! If you re-run this script, it will skip entries already filled.")


if __name__ == "__main__":
    main()
