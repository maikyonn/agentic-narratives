import pandas as pd
import json
import re

def extract_json(text):
    """Extract JSON-formatted turning points from text."""
    if pd.isna(text):
        return None
        
    # Look for JSON pattern with either single or double quotes
    # Pattern for single-line JSON
    json_pattern_single = r'\{(?:["\']tp\d+["\']:\s*\d+(?:\.\d+)?(?:,\s*["\']tp\d+["\']:\s*\d+(?:\.\d+)?)*)\}'
    # Pattern for multi-line formatted JSON with optional whitespace and newlines
    json_pattern_multi = r'\{\s*(?:["\']tp\d+["\']:\s*\d+(?:\.\d+)?(?:\s*,\s*["\']tp\d+["\']:\s*\d+(?:\.\d+)?)*\s*)\}'
    
    # Try both patterns
    match = re.search(json_pattern_single, text)
    if not match:
        match = re.search(json_pattern_multi, text)
    
    if not match:
        return None
        
    try:
        # Replace single quotes with double quotes for valid JSON
        json_str = match.group(0).replace("'", '"')
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def main():
    # Read the CSV file
    df = pd.read_csv('val_predictions.csv')
    
    # Extract JSON from both response columns
    df['extracted_tp_non_lora'] = df['non_lora_response'].apply(extract_json)
    df['extracted_tp_lora'] = df['lora_response'].apply(extract_json)
    
    # Save the updated dataframe
    df.to_csv('val_predictions.csv', index=False)

if __name__ == "__main__":
    main()
