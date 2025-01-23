#!/usr/bin/env python
import os
import json
import html
import argparse

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# ------------------------------------------------------------------------------
# 1. GLOBAL / DEFAULT CONFIG
# ------------------------------------------------------------------------------
CONFIG = {
    # File paths
    "COMBINED_NARRATIVES_FILE": "datasets/ground_truth.json",
    "OUTPUT_FILE_TP": "step2_model/llama3.1-405b-instruct-fp8/narratives_tp_reasoning.json",
    "OUTPUT_FILE_ARC": "step2_model/llama3.1-405b-instruct-fp8/narratives_arc_reasoning.json",
    "OUTPUT_FILE_ARC_TP": "step2_model/llama3.1-405b-instruct-fp8/narratives_arc_tp_reasoning.json",
    "ERROR_LOG_FILE": "local/logs/error_prompts.log",

    # LLM Configuration
    "LLM_API_KEY": "secret_narratives_652fe4188cfa48a89d8e941099004552.hs4awlXceqOliqZFCWKMV38luOnduxXr",      # Replace with actual API key if needed
    "LLM_MODEL": "llama3.1-405b-instruct-fp8",      # Replace with your actual model name
    "LLM_BASE_URL": "https://api.lambdalabs.com/v1",  # Replace with your actual base URL

    # LLM Generation Parameters
    "TEMPERATURE": 0.7,
    "MAX_TOKENS": 20000,
}

# ------------------------------------------------------------------------------
# 2. PROMPT TEMPLATES
# ------------------------------------------------------------------------------
PROMPT_TEMPLATES = {
    "tp": r"""
        ### INSTRUCTIONS
        You are a helpful assistant that identifies and explains turning points in a narrative. You are given:
        1) The story, broken down into numbered sentences.
        2) The definitions of each of the five turning points (Opportunity, Change of Plans, Point of No Return, Major Setback, Climax).
        3) Ground truth turning point indices for this story.

        ### TURNING POINT DEFINITIONS
        1. **Opportunity** – Introductory event that occurs after presenting the setting and background of the main characters.
        2. **Change of Plans** – Event where the main goal of the story is defined, starting the main action.
        3. **Point of No Return** – Event that pushes the main character(s) to fully commit to their goal.
        4. **Major Setback** – Event where things fall apart temporarily or permanently.
        5. **Climax** – Final event/resolution of the main story (the "biggest spoiler").

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
        1. Identify the protagonist or central group of characters in the story.
        2. Summarize each major event of the story, and explore how those events change the protagonist's condition.
        3. Using each ground truth turning point index, produce a detailed explanation of **why** that specific sentence qualifies as that turning point (based on the definitions). Typically turning points indicate a change in the protagonist's condition.
        4. Conclude your reasoning with a short summary that lists the turning points in order with their sentence indices.

        """,
    "arc": r"""### INSTRUCTIONS
    You are a writing teacher and your goal is to create examples on how to classify a story into one of several different story arc types based on the protagonist's condition throughout the story. Your goal is to teach it to a student, so you should explain your reasoning in a logical manner.

    ### STORY
    Below is the story, broken down into numbered sentences:

    {story}

    #### Story Arcs:
    -  **Rags to Riches:** Protagonist starts in a disadvantaged situation and ends in a much better one. The protagonist's condition improves from the first turning point to the last turning point. Example, 0 1 2 4 10.
    -  **Riches to Rags:** Protagonist starts in a high-status position but ends in a significantly lower state. The protagonist's condition worsens from the first turning point to the last turning point. Example, 10 9 8 6 0.
    -  **Man in a Hole:** Protagonist falls into a dilemma and finds a way out, ending better than at the beginning. The protagonist's condition improves from the first turning point to the last turning point. Example, 6 2 1 4 10.
    -  **Icarus:** Protagonist rises to success but then faces a drastic downfall. The protagonist's condition starts low at the first turning point, rises to its peak in the third turning point, and then falls to a low point at the last turning point. Example, 2 4 9 5 1.
    -  **Double Man in a Hole:** Protagonist faces two cycles of dilemma and recovery. Example, 6 1 5 1 10.
    -  **Cinderella:** Protagonist rises, faces a setback, and ultimately achieves a higher state. Example, 1 7 4 1 10.
    -  **Oedipus:** Protagonist starts high, falls, recovers, and then faces another significant downfall. Example, 10 4 7 9 1.

    ### STORY ARC CLASSIFICATION
    {story_arc}

    ### TASK
    1. Identify the protagonist in the story, and identify 5 major events in the story.
    2. At each event, describe the protagonist's state and how it changed relative to the previous events.
    3. Classify the story arc type based on the protagonist's condition thoughout and explain your reasoning.
    4. End by simply stating the determined story arc type.

""",
    "arc-tp": r"""### INSTRUCTIONS
    You are a writing teacher and your goal is to create examples on how to classify a story into one of several different story arc types based on the protagonist's condition at each turning point. Your goal is to teach it to a student, so you should explain your reasoning in a logical manner.

    ### STORY
    Below is the story, broken down into numbered sentences:

    {story}

    ### GROUND TRUTH TURNING POINTS
    - Opportunity (tp1): {tp1} - Introductory event that occurs after presenting the setting and background of the main characters.
    - Change of Plans (tp2): {tp2} - Event where the main goal of the story is defined, starting the main action.
    - Point of No Return (tp3): {tp3} - Event that pushes the main character(s) to fully commit to their goal.
    - Major Setback (tp4): {tp4} - Event where things fall apart temporarily or permanently.
    - Climax (tp5): {tp5} - Final event/resolution of the main story (the "biggest spoiler").

    #### Story Arcs:
    -  **Rags to Riches:** Protagonist starts in a disadvantaged situation and ends in a much better one. The protagonist's condition improves from the first turning point to the last turning point. Example, 0 1 2 4 10.
    -  **Riches to Rags:** Protagonist starts in a high-status position but ends in a significantly lower state. The protagonist's condition worsens from the first turning point to the last turning point. Example, 10 9 8 6 0.
    -  **Man in a Hole:** Protagonist falls into a dilemma and finds a way out, ending better than at the beginning. The protagonist's condition improves from the first turning point to the last turning point. Example, 6 2 1 4 10.
    -  **Icarus:** Protagonist rises to success but then faces a drastic downfall. The protagonist's condition starts low at the first turning point, rises to its peak in the third turning point, and then falls to a low point at the last turning point. Example, 2 4 9 5 1.
    -  **Double Man in a Hole:** Protagonist faces two cycles of dilemma and recovery. Example, 6 1 5 1 10.
    -  **Cinderella:** Protagonist rises, faces a setback, and ultimately achieves a higher state. Example, 1 7 4 1 10.
    -  **Oedipus:** Protagonist starts high, falls, recovers, and then faces another significant downfall. Example, 10 4 7 9 1.

    ### STORY ARC CLASSIFICATION
    {story_arc}

    ### TASK
    1. Identify the protagonist in the story
    2. At the sentence indicated by the first turning point, state the sentence again, and describe the protagonist's state.
    3. Identify the second turning point and state the sentence corresponding to that turning point, and describe how it changed relative to the first turning point.
    4. Similarly, identify the third turning point and state the sentence corresponding to that turning point, and describe how it changed relative to the second turning point.
    5. Then, identify the fourth turning point and state the sentence corresponding to that turning point, and describe how it changed relative to the third turning point.
    6. Finally, identify the fifth turning point and state the sentence corresponding to that turning point, and describe how it changed relative to the fourth turning point.
    7. At every turning point, approximate the protagonist's condition in the story as a number from 0 to 10, where 0 is the worst possible condition and 10 is the best possible condition. Put these 5 numbers describing the protagonist's condition in a list in chronological order.
    8. Classify the story arc type based on the protagonist's condition list and explain your reasoning.
    9. End by simply stating the determined story arc type.

"""
}

# ------------------------------------------------------------------------------
# 3. LLM CLIENT AND UTILS
# ------------------------------------------------------------------------------

def get_llm_client(api_key, model_name, base_url, temperature, max_tokens):
    """
    Returns a ChatOpenAI client based on configuration.
    Adjust the parameter names if using a different LLM provider.
    """
    return ChatOpenAI(
        openai_api_key=api_key,
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens
    )

def load_json(file_path):
    """
    Loads a JSON file and returns its content.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    """
    Saves data to a JSON file with pretty formatting.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def append_error_log(error_log_path, narrative_id, prompt, error_message):
    """
    Appends error details to the error log file.
    """
    with open(error_log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"Narrative ID: {narrative_id}\nPrompt:\n{prompt}\nError: {error_message}\n{'-'*50}\n")

def ensure_directory_exists(file_path):
    """
    Creates the directory path if it doesn't exist.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def create_numbered_story(synopsis_list):
    """
    Convert a list of sentences into a single string, 
    each sentence prefixed with its index for clarity.
    """
    return "\n".join(f"{i+1}) {s}" for i, s in enumerate(synopsis_list))

def build_prompt(mode, prompt_template, numbered_story, turning_points, story_arc):
    """
    Build a formatted prompt for either 'tp' (turning points), 'arc', or 'arc-tp' (story arc using turning points) mode.
    """
    # Extract turning point values (or use 'N/A' if missing)
    tp_values = {
        'tp1': turning_points.get('tp1', 'N/A'),
        'tp2': turning_points.get('tp2', 'N/A'),
        'tp3': turning_points.get('tp3', 'N/A'),
        'tp4': turning_points.get('tp4', 'N/A'),
        'tp5': turning_points.get('tp5', 'N/A'),
    }

    # Build the prompt based on mode
    if mode in ["arc", "arc-tp"]:
        return prompt_template.format(
            story=numbered_story,
            story_arc=story_arc,
            **tp_values
        )
    else:
        # 'tp' mode
        return prompt_template.format(
            story=numbered_story,
            **tp_values
        )

def invoke_llm_with_prompt(llm_client, narrative_id, prompt_str, error_log_path):
    """
    Safely invokes the LLM with the given prompt and returns the text content or an empty string on error.
    """
    try:
        response = llm_client.invoke(prompt_str)
        # Some clients return the result in .content, others directly as string
        return response.content.strip() if hasattr(response, 'content') else str(response).strip()
    except Exception as e:
        error_msg = f"Error generating reasoning for Narrative ID {narrative_id}: {str(e)}"
        append_error_log(error_log_path, narrative_id, prompt_str, error_msg)
        print(error_msg)
        return ""

def build_train_data(mode, synopsis, turning_points=None, story_arc=None, reasoning_text=None):
    if mode == "tp":
        instruction = """### INSTRUCTIONS
        You are a helpful assistant that identifies and explains turning points in a narrative. You are given:
        - The story, broken down into numbered sentences.
        - The definitions of each of the five turning points (Opportunity, Change of Plans, Point of No Return, Major Setback, Climax).
        - Ground truth turning point indices for this story.
        
        ### TURNING POINT DEFINITIONS
        - **Opportunity** – Introductory event that occurs after presenting the setting and background of the main characters.
        - **Change of Plans** – Event where the main goal of the story is defined, starting the main action.
        - **Point of No Return** – Event that pushes the main character(s) to fully commit to their goal.
        - **Major Setback** – Event where things fall apart temporarily or permanently.
        - **Climax** – Final event/resolution of the main story (the "biggest spoiler").
    
        ### STORY
        Below is the story, broken down into numbered sentences:
        {numbered_story}

        Please use the provided definitions and the sentence indices to produce a step-by-step reasoning ("chain of thought") explaining **why** each sentence index corresponds to the turning point category. End with a concise summary that reiterates the turning points in order.

        ### RESPONSE
        {reasoning_text}
        """
        
        key_name = "tp_train_data"
    elif mode == "arc":
        instruction = """
            ### INSTRUCTIONS 
            Analyze the story and classify it into one of the following story arc types based on the protagonist's condition at each major event. 
            Explain your reasoning step by step and provide the determined story arc type.\n\n
            
            Story Arc Types:\n
            - Rags to Riches: Protagonist starts disadvantaged and ends better (e.g., 0→1→2→4→10)\n
            - Riches to Rags: Protagonist starts high and ends lower (e.g., 10→9→8→6→0)\n
            - Man in a Hole: Protagonist falls into trouble but recovers (e.g., 6→2→1→4→10)\n
            - Icarus: Protagonist rises then falls dramatically (e.g., 2→4→9→5→1)\n
            - Double Man in a Hole: Two cycles of fall and recovery (e.g., 6→2→1→4→10)\n
            - Cinderella: Rise, setback, ultimate triumph (e.g., 1→7→4→1→10)\n
            - Oedipus: Start high, fall, recover, final fall (e.g., 10→4→7→9→1)
    
            ### SYNOPSIS
            {numbered_story}
    
            ### RESPONSE
            {reasoning_text}
        """
    elif mode == "arc-tp":
        instruction = """
            Analyze the story and classify it into one of the following story arc types based on the protagonist's condition at each turning point. 
            Explain your reasoning step by step and provide the determined story arc type.\n\n
            Story Arc Types:\n
            - Rags to Riches: Protagonist starts disadvantaged and ends better (e.g., 0→1→2→4→10)\n
            - Riches to Rags: Protagonist starts high and ends lower (e.g., 10→9→8→6→0)\n
            - Man in a Hole: Protagonist falls into trouble but recovers (e.g., 6→2→1→4→10)\n
            - Icarus: Protagonist rises then falls dramatically (e.g., 2→4→9→5→1)\n
            - Double Man in a Hole: Two cycles of fall and recovery (e.g., 6→2→1→4→10)\n
            - Cinderella: Rise, setback, ultimate triumph (e.g., 1→7→4→1→10)\n
            - Oedipus: Protagonist starts high, falls, recover, final fall (e.g., 10→4→7→9→1)
            
            ### GROUND TRUTH TURNING POINTS
            - Opportunity (tp1): {tp1} - Introductory event that occurs after presenting the setting and background of the main characters.
            - Change of Plans (tp2): {tp2} - Event where the main goal of the story is defined, starting the main action.
            - Point of No Return (tp3): {tp3} - Event that pushes the main character(s) to fully commit to their goal.
            - Major Setback (tp4): {tp4} - Event where things fall apart temporarily or permanently.
            - Climax (tp5): {tp5} - Final event/resolution of the main story (the “biggest spoiler”).
    
            ### SYNOPSIS
            {numbered_story}
    
            ### RESPONSE
            {reasoning_text}
        """

    numbered_story = create_numbered_story(synopsis)

    if mode == "arc":
        train_data_str = instruction.format(
            numbered_story=numbered_story,
            reasoning_text=reasoning_text
        )
    else:
        train_data_str = instruction.format(
            numbered_story=numbered_story,
            tp1=turning_points.get('tp1', 'N/A'),
            tp2=turning_points.get('tp2', 'N/A'),
            tp3=turning_points.get('tp3', 'N/A'),
            tp4=turning_points.get('tp4', 'N/A'),
            tp5=turning_points.get('tp5', 'N/A'),
            story_arc=story_arc,
            reasoning_text=reasoning_text
        )

    return train_data_str

# ------------------------------------------------------------------------------
# 4. MAIN PROCESSING LOGIC
# ------------------------------------------------------------------------------

def generate_reasoning_data(
    narratives,
    llm_client,
    prompt_template,
    output_file,
    error_log_path,
    mode,
    overwrite=False
):
    """
    Generates reasoning data for each narrative by invoking the LLM with the appropriate prompts.
    Adds the reasoning to each narrative under the specified field and saves incrementally.
    """
    total_narratives = len(narratives)
    processed_count = 0
    skipped_count = 0

    # Decide which field we're populating based on mode
    if mode == "tp":
        reasoning_field = 'tp_pred_reasoning'
    elif mode == "arc":
        reasoning_field = 'arc_pred_reasoning'
    elif mode == "arc-tp":
        reasoning_field = 'arc_tp_pred_reasoning'
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for idx, narrative in enumerate(narratives, start=1):
        narrative_id = narrative.get('narrative_id', f'Unknown_{idx}')
        print(f"\nProcessing narrative {idx}/{total_narratives} - ID: {narrative_id}")

        # 1) Skip logic if we do NOT want to overwrite existing reasoning
        if not overwrite and reasoning_field in narrative and narrative[reasoning_field]:
            print(f"Skipping ID {narrative_id} (already has '{reasoning_field}').")
            skipped_count += 1
            continue

        # 2) Extract fields and build the story
        synopsis = narrative.get('synopsis', [])
        turning_points = narrative.get('turning_points', {})
        story_arc = narrative.get('arc_label', 'N/A')
        numbered_story = create_numbered_story(synopsis)

        # 3) Build the prompt
        try:
            prompt_str = build_prompt(
                mode=mode,
                prompt_template=prompt_template,
                numbered_story=numbered_story,
                turning_points=turning_points,
                story_arc=story_arc
            )
        except KeyError as e:
            error_msg = f"Missing placeholder in prompt: {str(e)}"
            append_error_log(error_log_path, narrative_id, prompt_template, error_msg)
            print(error_msg)
            continue

        # 4) Print the final prompt for debug
        print("\nPrompt:")
        print("=" * 80)
        print(prompt_str)
        print("=" * 80)

        # 5) Invoke the LLM
        reasoning_text = invoke_llm_with_prompt(llm_client, narrative_id, prompt_str, error_log_path)
        if not reasoning_text:
            # If we got an error or empty response, skip saving
            continue

        # 6) Print the generated output for clarity
        print("\nGenerated Reasoning:")
        print("-" * 80)
        print(reasoning_text)
        print("-" * 80)

        # 7) Save the reasoning to the narrative
        narrative[reasoning_field] = reasoning_text

        # 8) Save the updated narratives to the output file incrementally
        save_json(narratives, output_file)

        processed_count += 1
        print(f"Successfully generated reasoning for Narrative ID {narrative_id} and saved progress.")

    print(f"\nReasoning generation complete. Processed: {processed_count}, Skipped: {skipped_count}.")

def transform_reasoning_to_training_data(
    narratives,
    output_file,
    mode
):
    """
    Transforms the generated reasoning data into training data for each narrative.
    Adds the training data to each narrative under the specified field and saves incrementally.
    """
    for idx, narrative in enumerate(narratives, start=1):
        synopsis = narrative.get('synopsis', [])
        turning_points = narrative.get('turning_points', {})
        story_arc = narrative.get('arc_label', 'N/A')
        if mode == "tp":
            reasoning_text = narrative.get('tp_pred_reasoning')
        elif mode == "arc":
            reasoning_text = narrative.get('arc_pred_reasoning')
        elif mode == "arc-tp":
            reasoning_text = narrative.get('arc_tp_pred_reasoning')
        else:
            reasoning_text = None  # Unknown mode

        if not reasoning_text:
            continue  # Skip if reasoning is not available

        # Build training data
        training_text = build_train_data(
            mode=mode,
            synopsis=synopsis,
            turning_points=turning_points,
            story_arc=story_arc,
            reasoning_text=reasoning_text
        )

        # Assign the training data to the appropriate field
        if mode == 'arc-tp':
            narrative['arc_tp_training_text'] = training_text
        elif mode == 'arc':
            narrative['arc_training_text'] = training_text
        else:  # 'tp'
            narrative['tp_training_text'] = training_text

        # Save the updated narratives to the output file incrementally
        save_json(narratives, output_file)

        print(f"Transformed training data for Narrative ID {narrative.get('narrative_id', 'Unknown')} and saved progress.")

    print("Training data transformation complete.")

def process_narratives(
    narratives,
    llm_client,
    prompt_template,
    output_file,
    error_log_path,
    mode,
    overwrite=False
):
    """
    Processes narratives in two phases:
    1. Generates reasoning data.
    2. Transforms reasoning data into training data.
    Saves after each narrative is processed to allow multiple separate runs.
    """
    # Phase 1: Generate Reasoning Data
    generate_reasoning_data(
        narratives=narratives,
        llm_client=llm_client,
        prompt_template=prompt_template,
        output_file=output_file,
        error_log_path=error_log_path,
        mode=mode,
        overwrite=overwrite
    )

    # Phase 2: Transform Reasoning to Training Data
    transform_reasoning_to_training_data(
        narratives=narratives,
        output_file=output_file,
        mode=mode
    )

    print(f"All eligible narratives have been processed and saved to {output_file} successfully.")

# ------------------------------------------------------------------------------
# 5. MAIN ENTRY POINT
# ------------------------------------------------------------------------------

def initialize_output_file(source_file, output_file):
    """
    Creates the output JSON file if it doesn't exist by copying from source.
    Returns the loaded JSON data.
    """
    if not os.path.exists(output_file):
        # Load data from source file
        source_data = load_json(source_file)
        # Create the output file with the source data
        save_json(source_data, output_file)
        print(f"Initialized new output file: {output_file}")
        return source_data
    else:
        # Load existing output file
        return load_json(output_file)

def main():
    parser = argparse.ArgumentParser(
        description="Script to generate LLM outputs (chain-of-thought or story arc) for narratives."
    )
    parser.add_argument(
        "--mode",
        choices=["tp", "arc-tp", "arc"],
        default="tp",
        help="Choose which prompt template to use: 'tp' for turning-point CoT, 'arc-tp' for story arc using turning points, 'arc' for story arc classification."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, will overwrite existing entries instead of skipping them."
    )
    parser.add_argument(
        "--training-data-only",
        action="store_true",
        help="If set, will only generate training data for narratives that already have reasoning but are missing training data."
    )
    args = parser.parse_args()

    # Pick the correct output file and prompt template based on mode
    if args.mode == "tp":
        output_file = CONFIG["OUTPUT_FILE_TP"]
        prompt_template = PROMPT_TEMPLATES["tp"]
    elif args.mode == "arc":
        output_file = CONFIG["OUTPUT_FILE_ARC"]
        prompt_template = PROMPT_TEMPLATES["arc"]
    elif args.mode == "arc-tp":
        output_file = CONFIG["OUTPUT_FILE_ARC_TP"]
        prompt_template = PROMPT_TEMPLATES["arc-tp"]
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Ensure output and log directories exist
    ensure_directory_exists(output_file)
    ensure_directory_exists(CONFIG["ERROR_LOG_FILE"])

    # Initialize or load the output file
    narratives = initialize_output_file(CONFIG["COMBINED_NARRATIVES_FILE"], output_file)

    if args.training_data_only:
        # Only transform reasoning to training data
        transform_reasoning_to_training_data(
            narratives=narratives,
            output_file=output_file,
            mode=args.mode
        )
    else:
        # Get LLM client
        llm_client = get_llm_client(
            api_key=CONFIG["LLM_API_KEY"],
            model_name=CONFIG["LLM_MODEL"],
            base_url=CONFIG["LLM_BASE_URL"],
            temperature=CONFIG["TEMPERATURE"],
            max_tokens=CONFIG["MAX_TOKENS"]
        )

        # Process the narratives (both reasoning and training data)
        process_narratives(
            narratives=narratives,
            llm_client=llm_client,
            prompt_template=prompt_template,
            output_file=output_file,
            error_log_path=CONFIG["ERROR_LOG_FILE"],
            mode=args.mode,
            overwrite=args.overwrite
        )

    print("\nAll eligible narratives have been processed and saved successfully.")

if __name__ == "__main__":
    main()