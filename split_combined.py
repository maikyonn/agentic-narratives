import json
import random
from typing import List, Dict, Any

def load_narratives(file_path: str) -> List[Dict[str, Any]]:
    """Load narratives from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def split_by_ground_truth(narratives: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split narratives into those with and without turning points/arc labels."""
    with_tp = []
    without_tp = []
    
    for narrative in narratives:
        # Check if narrative has turning points and arc label
        has_ground_truth = (
            narrative.get('turning_points') is not None and 
            narrative.get('arc_label') is not None and
            narrative.get('tp_pred_reasoning') is not None
        )
        
        if has_ground_truth:
            with_tp.append(narrative)
        else:
            without_tp.append(narrative)
    
    return with_tp, without_tp

def train_val_test_split(
    data: List[Dict[str, Any]], 
    train_ratio: float = 0.75,
    val_ratio: float = 0.15
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split data into training, validation, and test sets."""
    # Shuffle data
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    # Calculate split indices
    train_idx = int(len(shuffled) * train_ratio)
    val_idx = train_idx + int(len(shuffled) * val_ratio)
    
    return shuffled[:train_idx], shuffled[train_idx:val_idx], shuffled[val_idx:]

def save_json(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load narratives
    narratives = load_narratives('combined_narratives_with_reasoning.json')
    
    # Split into ground truth and non-ground truth
    ground_truth, non_ground_truth = split_by_ground_truth(narratives)
    
    # Create train/val/test split from ground truth
    train_data, val_data, test_data = train_val_test_split(ground_truth)
    
    # Save all splits
    save_json(ground_truth, 'ground_truth.json')
    save_json(non_ground_truth, 'non_ground_truth.json')
    save_json(train_data, 'train.json')
    save_json(val_data, 'val.json')
    save_json(test_data, 'test.json')
    
    # Print statistics
    print(f"Total narratives: {len(narratives)}")
    print(f"Ground truth narratives: {len(ground_truth)}")
    print(f"Non-ground truth narratives: {len(non_ground_truth)}")
    print(f"Training set size: {len(train_data)} ({len(train_data)/len(ground_truth)*100:.1f}%)")
    print(f"Validation set size: {len(val_data)} ({len(val_data)/len(ground_truth)*100:.1f}%)")
    print(f"Test set size: {len(test_data)} ({len(test_data)/len(ground_truth)*100:.1f}%)")

if __name__ == "__main__":
    main()
