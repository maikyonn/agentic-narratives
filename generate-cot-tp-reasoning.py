import json
import os
import html
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# ---------------------------------
# CONFIGURATION
# ---------------------------------

# File paths
COMBINED_NARRATIVES_FILE = 'data_release/combined_narratives.json'
OUTPUT_FILE = 'combined_narratives_with_reasoning.json'
ERROR_LOG_FILE = 'error_prompts.log'

# LLM Configuration
LLM_API_KEY = "ollama"  # Replace with your actual API key if required
LLM_MODEL = "llama3.3"  # Replace with your actual model name
LLM_BASE_URL = "http://localhost:11434/v1"  # Replace with your actual base URL

# ---------------------------------
# INITIALIZE LLM CLIENT
# ---------------------------------

# Initialize the LLM client using LangChain's ChatOpenAI
llm = ChatOpenAI(
    openai_api_key=LLM_API_KEY,  # Adjust parameter name if using a different provider
    model=LLM_MODEL,
    base_url=LLM_BASE_URL,
    temperature=0.7,  # Adjust as needed
    max_tokens=1500,  # Adjust based on your requirements
)

# ---------------------------------
# DEFINE PROMPT TEMPLATE
# ---------------------------------

narrative_template = """### INSTRUCTIONS
You are a helpful assistant that identifies and explains turning points in a narrative. You are given:
1) The story, broken down into numbered sentences.
2) The definitions of each of the five turning points (Opportunity, Change of Plans, Point of No Return, Major Setback, Climax).
3) Ground truth turning point indices for this story.

Please use the provided definitions and the sentence indices to produce a step-by-step reasoning (“chain of thought”) explaining **why** each sentence index corresponds to the turning point category. End with a concise summary that reiterates the turning points in order.

### TURNING POINT DEFINITIONS
1. **Opportunity** – Introductory event that occurs after presenting the setting and background of the main characters.
2. **Change of Plans** – Event where the main goal of the story is defined, starting the main action.
3. **Point of No Return** – Event that pushes the main character(s) to fully commit to their goal.
4. **Major Setback** – Event where things fall apart temporarily or permanently.
5. **Climax** – Final event/resolution of the main story (the “biggest spoiler”).

### STORY
Below is the story, broken down into numbered sentences:

{story}

### GROUND TRUTH TURNING POINTS
- Opportunity (tp1): {tp1}
- Change of Plans (tp2): {tp2}
- Point of No Return (tp3): {tp3}
- Major Setback (tp4): {tp4}
- Climax (tp5): {tp5}

### TASK
1. Review the story.
2. Summarize each segment of the story, and explore the event of each character individually and how they change or are affected throughout the story.
3. Using each ground truth turning point index, produce a detailed chain-of-thought explanation of **why** that specific sentence qualifies as that turning point (based on the definitions).
4. Conclude your reasoning with a short summary that lists the turning points in order with their sentence indices.

### RESPONSE FORMAT
**Turning Point Explanation**:
(Explain your reasoning about each turning point in a step-by-step manner referencing the relevant sentence(s).)
"""

# ---------------------------------
# LOAD AND SAVE JSON FUNCTIONS
# ---------------------------------

def load_json(file_path):
    """
    Loads a JSON file and returns its content.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            print(f"Successfully loaded {file_path}")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {e}")

def save_json(data, file_path):
    """
    Saves data to a JSON file with pretty formatting.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved data to {file_path}")

def append_error_log(narrative_id, prompt, error_message):
    """
    Appends error details to the error log file.
    """
    with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as log_file:
        log_file.write(f"Narrative ID: {narrative_id}\nPrompt:\n{prompt}\nError: {error_message}\n{'-'*50}\n")

# ---------------------------------
# PROCESS NARRATIVES
# ---------------------------------

def process_narratives(narratives, llm_client, prompt_template, output_file):
    """
    Processes each narrative by generating tp_pred_reasoning using the LLM.
    Saves the JSON after processing each narrative.
    Skips narratives that already have a non-null tp_pred_reasoning.
    
    :param narratives: List of narrative dictionaries.
    :param llm_client: Initialized LLM client.
    :param prompt_template: Prompt template string.
    :param output_file: Path to save the updated JSON after each processing.
    :return: None
    """
    total_narratives = len(narratives)
    processed_count = 0
    skipped_count = 0

    for idx, narrative in enumerate(narratives, 1):
        narrative_id = narrative.get('narrative_id', 'Unknown ID')
        print(f"\nProcessing narrative {idx}/{total_narratives} - ID: {narrative_id}")

        # Check if 'tp_pred_reasoning' already exists and is not null
        if 'tp_pred_reasoning' in narrative and narrative['tp_pred_reasoning']:
            print(f"Skipping narrative ID {narrative_id} as it already has 'tp_pred_reasoning'.")
            skipped_count += 1
            continue

        # Extract necessary fields
        synopsis = narrative.get('synopsis', [])
        turning_points = narrative.get('turning_points', {})

        if not synopsis:
            print(f"Warning: No synopsis found for narrative ID {narrative_id}. Skipping.")
            skipped_count += 1
            continue

        if not turning_points:
            print(f"Warning: No turning points found for narrative ID {narrative_id}. Skipping.")
            skipped_count += 1
            continue

        # Number the synopsis sentences
        numbered_story = "\n".join([f"{i+1}) {sentence}" for i, sentence in enumerate(synopsis)])

        # Prepare the prompt by filling in the template
        prompt = prompt_template.format(
            story=numbered_story,
            tp1=turning_points.get('tp1', 'N/A'),
            tp2=turning_points.get('tp2', 'N/A'),
            tp3=turning_points.get('tp3', 'N/A'),
            tp4=turning_points.get('tp4', 'N/A'),
            tp5=turning_points.get('tp5', 'N/A')
        )

        try:
            # Invoke the LLM with the prompt
            response = llm_client.invoke(prompt)

            # Depending on the LLM's response structure, adjust accordingly
            # Assuming response.content contains the reasoning text
            if hasattr(response, 'content'):
                reasoning = response.content.strip()
            elif isinstance(response, str):
                reasoning = response.strip()
            else:
                reasoning = str(response).strip()

            # Add the reasoning to the narrative
            narrative['tp_pred_reasoning'] = reasoning

            print(f"Successfully processed narrative ID {narrative_id}.")
            processed_count += 1

        except Exception as e:
            print(f"Error processing narrative ID {narrative_id}: {e}")
            # Log the error with narrative ID and prompt
            append_error_log(narrative_id, prompt, str(e))
            # Assign None or a default value to indicate failure
            narrative['tp_pred_reasoning'] = None

        # Save the updated narratives to the output file after each processing
        try:
            save_json(narratives, output_file)
        except Exception as save_error:
            print(f"Error saving to {output_file}: {save_error}")
            # Optionally, log the save error
            append_error_log(narrative_id, prompt, f"Save Error: {save_error}")
            # Decide whether to continue or halt; here, we'll continue
            continue

    print(f"\nProcessing complete. Total processed: {processed_count}, Total skipped: {skipped_count}.")

# ---------------------------------
# MAIN FUNCTION
# ---------------------------------

def main():
    # Check if OUTPUT_FILE exists. If so, load it to resume processing; else, load COMBINED_NARRATIVES_FILE
    if os.path.exists(OUTPUT_FILE):
        print(f"Loading existing output file: {OUTPUT_FILE}")
        try:
            narratives = load_json(OUTPUT_FILE)
        except Exception as e:
            print(f"Failed to load {OUTPUT_FILE}: {e}")
            return
    else:
        print(f"Loading combined narratives file: {COMBINED_NARRATIVES_FILE}")
        try:
            narratives = load_json(COMBINED_NARRATIVES_FILE)
        except Exception as e:
            print(f"Failed to load {COMBINED_NARRATIVES_FILE}: {e}")
            return

    # Check if narratives is a list
    if not isinstance(narratives, list):
        print(f"Error: Expected a list of narratives in the loaded JSON.")
        return

    # Process narratives to add tp_pred_reasoning
    process_narratives(narratives, llm, narrative_template, OUTPUT_FILE)

    print("\nAll eligible narratives have been processed and saved successfully.")

if __name__ == "__main__":
    main()