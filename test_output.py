from finetune import load_json_data, format_example, tokenize_function
from transformers import AutoTokenizer

def main():
    # Load the same model tokenizer
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token if needed

    # Load first data point
    json_path = "ground_truth.json"
    raw_data = load_json_data(json_path)
    formatted_example = format_example(raw_data[0])

    # Show the formatted example before tokenization
    print("=== Formatted Example ===")
    print("\n--- Instruction ---")
    print(formatted_example["instruction"])
    print("\n--- Input ---")
    print(formatted_example["input"])
    print("\n--- Output ---")
    print(formatted_example["output"])

    # Show tokenized version
    print("\n=== Tokenized Output ===")
    tokenized = tokenize_function(formatted_example, tokenizer)
    
    # Decode back to text to see how it looks
    print("\n--- Decoded Text ---")
    decoded_text = tokenizer.decode(tokenized["input_ids"])
    print(decoded_text)

    # Show token statistics
    print("\n--- Token Statistics ---")
    print(f"Number of tokens: {len(tokenized['input_ids'])}")
    print(f"Input shape: {tokenized['input_ids'].shape if hasattr(tokenized['input_ids'], 'shape') else len(tokenized['input_ids'])}")

if __name__ == "__main__":
    main() 