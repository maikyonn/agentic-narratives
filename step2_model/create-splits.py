import json
import random
import os

def split_data(input_file, train_ratio=0.8):
    """
    Split JSON data into training and validation sets.
    
    Args:
        input_file (str): Path to input JSON file
        train_ratio (float): Ratio of data to use for training (default 0.8)
    """
    # Read input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get total number of samples
    n_samples = len(data)
    n_train = int(n_samples * train_ratio)
    
    # Randomly shuffle indices
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    # Split into train and validation sets
    train_data = [data[i] for i in indices[:n_train]]
    val_data = [data[i] for i in indices[n_train:]]
    
    # Create output filenames
    base_name = os.path.splitext(input_file)[0]
    train_file = f"{base_name}_train.json"
    val_file = f"{base_name}_val.json"
    
    # Write train and validation files
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)
        
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=4)
        
    print(f"Split {n_samples} samples into:")
    print(f"Training: {len(train_data)} samples saved to {train_file}")
    print(f"Validation: {len(val_data)} samples saved to {val_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python create-splits.py <input_json_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    split_data(input_file)
