import os
import json
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import random

def publish_dataset(json_path: str, dataset_name: str, test_size: float = 0.1, seed: int = 42):
    """
    Publish a dataset to the Hugging Face Hub from a JSON file containing instruction/input/output data.
    Creates train/test splits.
    
    Args:
        json_path: Path to JSON file containing the data
        dataset_name: Name to publish dataset as on HF Hub
        test_size: Fraction of data to use for test set (default: 0.1)
        seed: Random seed for reproducibility (default: 42)
    """
    # Load the JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract relevant fields
    dataset_dicts = []
    for item in data:
        dataset_dicts.append({
            'instruction': item['instruction'],
            'input': item['input'], 
            'output': item['output']
        })
    
    # Set random seed
    random.seed(seed)
    
    # Randomly shuffle data
    random.shuffle(dataset_dicts)
    
    # Calculate split point
    split_idx = int(len(dataset_dicts) * (1 - test_size))
    
    # Split into train/test
    train_dicts = dataset_dicts[:split_idx]
    test_dicts = dataset_dicts[split_idx:]
    
    # Create datasets
    train_dataset = Dataset.from_list(train_dicts)
    test_dataset = Dataset.from_list(test_dicts)
    
    # Combine into DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    # Push to hub
    dataset_dict.push_to_hub(dataset_name)
    print(f"Published dataset {dataset_name} to Hugging Face Hub")
    print(f"Train size: {len(train_dicts)}, Test size: {len(test_dicts)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', help='Path to input JSON file')
    parser.add_argument('dataset_name', help='Name to publish dataset as on HF Hub')
    parser.add_argument('--test-size', type=float, default=0.1,
                      help='Fraction of data to use for test set (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    args = parser.parse_args()
    
    publish_dataset(args.json_path, args.dataset_name, args.test_size, args.seed)
