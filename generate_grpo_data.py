import os
import json
import argparse

def create_numbered_story(synopsis_list):
    """
    Convert a list of sentences into a single string, 
    each sentence prefixed with its index for clarity.
    """
    return "\n".join(f"{i+1}) {s}" for i, s in enumerate(synopsis_list))

def build_train_data(synopsis, turning_points, story_arc, narrative_id=None):
    """Builds the training data with instruction, input, and arc_label."""
    numbered_story = create_numbered_story(synopsis)
    
    instruction = """### INSTRUCTIONS\nAnalyze the story and classify it into one of the following story arc types based on the protagonist's condition at each turning point.
        Story Arc Types:
        - Rags to Riches: Protagonist starts disadvantaged and ends better (e.g., 0→1→2→4→10)
        - Riches to Rags: Protagonist starts high and ends lower (e.g., 10→9→8→6→0)
        - Man in a Hole: Protagonist falls into trouble but recovers (e.g., 6→2→1→4→10)
        - Icarus: Protagonist rises then falls dramatically (e.g., 2→4→9→5→1)
        - Double Man in a Hole: Two cycles of fall and recovery (e.g., 6→2→7→4→10)
        - Cinderella: Rise, setback, ultimate triumph (e.g., 1→7→4→1→10)
        - Oedipus: Start high, fall, recover, final fall (e.g., 10→4→7→9→1)
"""
    input_text = f""" Story, broken down into numbered sentences:
{numbered_story}"""
    ground_truth_turning_points = f"""Ground Truth Turning Points And their corresponding sentences:
    - Opportunity (tp1): Sentence {turning_points.get('tp1', 'N/A')} - Introductory event that occurs after presenting the setting and background of the main characters.
    - Change of Plans (tp2): Sentence {turning_points.get('tp2', 'N/A')} - Event where the main goal of the story is defined, starting the main action.
    - Point of No Return (tp3): Sentence {turning_points.get('tp3', 'N/A')} - Event that pushes the main character(s) to fully commit to their goal.
    - Major Setback (tp4): Sentence {turning_points.get('tp4', 'N/A')} - Event where things fall apart temporarily or permanently.
    - Climax (tp5): Sentence {turning_points.get('tp5', 'N/A')} - Final event/resolution of the main story (the "biggest spoiler")."""

    return {
        "instruction": instruction,
        "story": input_text,
        "turning_points": ground_truth_turning_points,
        "json_turning_points": turning_points,
        "arc_label": story_arc
    }

def transform_reasoning_to_training_data(input_file, output_dir):
    """
    Transforms existing reasoning data into training data with instruction, input, and arc_label.
    
    Args:
        input_file: Path to the input JSON file containing narratives with reasoning
        output_dir: Directory to save the output file
    """
    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        narratives = json.load(f)

    # Prepare output data
    training_data = []

    # Process each narrative
    for narrative in narratives:
        training_example = build_train_data(
            synopsis=narrative.get('synopsis', []),
            turning_points=narrative.get('turning_points', {}),
            story_arc=narrative.get('arc_label', 'N/A'),
            narrative_id=narrative.get('narrative_id')
        )
        training_data.append(training_example)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the training data
    output_file = os.path.join(output_dir, "arc_tp_data.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=4, ensure_ascii=False)

    print(f"Processed {len(training_data)} narratives")
    print(f"Saved training data to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Transform narrative data into training data format")
    parser.add_argument("input_file", help="Path to input JSON file containing narratives")
    parser.add_argument("output_dir", help="Directory to save the output file")
    
    args = parser.parse_args()
    transform_reasoning_to_training_data(args.input_file, args.output_dir)

if __name__ == "__main__":
    main()  