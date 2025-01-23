import os
import json
import argparse

def create_numbered_story(synopsis_list):
    """
    Convert a list of sentences into a single string, 
    each sentence prefixed with its index for clarity.
    """
    return "\n".join(f"{i+1}) {s}" for i, s in enumerate(synopsis_list))

def build_train_data(mode, synopsis, turning_points=None, story_arc=None, reasoning_text=None):
    """Builds the training data in Alpaca format based on the mode and input data."""
    numbered_story = create_numbered_story(synopsis)
    
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
        - **Climax** – Final event/resolution of the main story (the "biggest spoiler")."""
        input_text = f"""Story, broken down into numbered sentences:
            {numbered_story}

        Please use the provided definitions and the sentence indices to produce an explanation **why** each sentence index corresponds to the turning point category. End with a concise summary that reiterates the turning points in order.

        ### RESPONSE
        {reasoning_text}
        """
    elif mode == "arc":
        instruction = "Analyze the story and classify it into one of the following story arc types based on the protagonist's condition at each major event. Explain your reasoning step by step."
        input_text = f"""Story, broken down into numbered sentences:
{numbered_story}

Story Arc Types:
- Rags to Riches: Protagonist starts disadvantaged and ends better (e.g., 0→1→2→4→10)
- Riches to Rags: Protagonist starts high and ends lower (e.g., 10→9→8→6→0)
- Man in a Hole: Protagonist falls into trouble but recovers (e.g., 6→2→1→4→10)
- Icarus: Protagonist rises then falls dramatically (e.g., 2→4→9→5→1)
- Double Man in a Hole: Two cycles of fall and recovery (e.g., 6→2→7→4→10)
- Cinderella: Rise, setback, ultimate triumph (e.g., 1→7→4→1→10)
- Oedipus: Start high, fall, recover, final fall (e.g., 10→4→7→9→1)"""

    elif mode == "arc-tp":
        instruction = "Analyze the story and classify it into one of the story arc types based on the protagonist's condition at each turning point. Explain your reasoning step by step."
        input_text = f"""Story, broken down into numbered sentences:
{numbered_story}

Story Arc Types:
- Rags to Riches: Protagonist starts disadvantaged and ends better (e.g., 0→1→2→4→10)
- Riches to Rags: Protagonist starts high and ends lower (e.g., 10→9→8→6→0)
- Man in a Hole: Protagonist falls into trouble but recovers (e.g., 6→2→1→4→10)
- Icarus: Protagonist rises then falls dramatically (e.g., 2→4→9→5→1)
- Double Man in a Hole: Two cycles of fall and recovery (e.g., 6→2→7→4→10)
- Cinderella: Rise, setback, ultimate triumph (e.g., 1→7→4→1→10)
- Oedipus: Start high, fall, recover, final fall (e.g., 10→4→7→9→1)

Ground Truth Turning Points:
- Opportunity (tp1): {turning_points.get('tp1', 'N/A')} - Introductory event that occurs after presenting the setting and background of the main characters.
- Change of Plans (tp2): {turning_points.get('tp2', 'N/A')} - Event where the main goal of the story is defined, starting the main action.
- Point of No Return (tp3): {turning_points.get('tp3', 'N/A')} - Event that pushes the main character(s) to fully commit to their goal.
- Major Setback (tp4): {turning_points.get('tp4', 'N/A')} - Event where things fall apart temporarily or permanently.
- Climax (tp5): {turning_points.get('tp5', 'N/A')} - Final event/resolution of the main story (the "biggest spoiler")."""

    return {
        "instruction": instruction,
        "input": input_text,
        "output": reasoning_text
    }

def transform_reasoning_to_training_data(input_file, output_dir, mode):
    """
    Transforms existing reasoning data into Alpaca format training data.
    
    Args:
        input_file: Path to the input JSON file containing narratives with reasoning
        output_dir: Directory to save the output file
        mode: One of 'tp', 'arc', or 'arc-tp'
    """
    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        narratives = json.load(f)

    # Prepare output data
    alpaca_data = []
    
    # Map mode to reasoning field name
    reasoning_field = {
        "tp": "tp_pred_reasoning",
        "arc": "arc_pred_reasoning",
        "arc-tp": "arc_tp_pred_reasoning"
    }[mode]

    # Process each narrative
    for narrative in narratives:
        reasoning_text = narrative.get(reasoning_field)
        if not reasoning_text:
            continue

        # Build training data in Alpaca format
        alpaca_example = build_train_data(
            mode=mode,
            synopsis=narrative.get('synopsis', []),
            turning_points=narrative.get('turning_points', {}),
            story_arc=narrative.get('arc_label', 'N/A'),
            reasoning_text=reasoning_text
        )

        alpaca_data.append(alpaca_example)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the Alpaca format data
    output_file = os.path.join(output_dir, f"{mode}_data.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, indent=4, ensure_ascii=False)

    print(f"Processed {len(alpaca_data)} narratives")
    print(f"Saved Alpaca format training data to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Transform narrative reasoning data into training data format")
    parser.add_argument("input_file", help="Path to input JSON file containing narratives with reasoning")
    parser.add_argument("output_dir", help="Directory to save the output file")
    parser.add_argument(
        "--mode",
        choices=["tp", "arc-tp", "arc"],
        default="tp",
        help="Type of reasoning to process: 'tp' for turning points, 'arc-tp' for story arc with turning points, 'arc' for story arc"
    )
    
    args = parser.parse_args()
    transform_reasoning_to_training_data(args.input_file, args.output_dir, args.mode)

if __name__ == "__main__":
    main()  