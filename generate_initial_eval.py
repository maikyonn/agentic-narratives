#!/usr/bin/env python
import os
import json
import argparse

from openai import OpenAI # If you plan to use the OpenAI API. Otherwise, replace with your chosen client library.

def create_numbered_story(synopsis_list):
    """
    Convert a list of sentences into a single string,
    each sentence prefixed with its index for clarity.
    """
    return "\n".join(f"{i+1}) {s}" for i, s in enumerate(synopsis_list))

def main():
    parser = argparse.ArgumentParser(
        description="Script to query a model for turning points and story arc identification."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to be used (e.g., 'gpt-3.5-turbo')."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="datasets/ground_truth.json",
        help="Path to the input JSON file containing narratives."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="model_evaluation_output.json",
        help="Path to the output JSON file."
    )
    args = parser.parse_args()
    
    # Modify output filename to include model name and create directory if needed
    if args.output_file == "model_evaluation_output.json":  # only modify if using default
        output_dir = "./step1-landscape-eval"
        os.makedirs(output_dir, exist_ok=True)
        args.output_file = f"{output_dir}/{args.model_name}_evaluation_output.json"

    openai_api_key = "secret_narratives_652fe4188cfa48a89d8e941099004552.hs4awlXceqOliqZFCWKMV38luOnduxXr"
    openai_api_base = "https://api.lambdalabs.com/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )


    # If you're using the OpenAI API, make sure your key is set as an environment variable
    # or set openai.api_key here directly (not recommended to hardcode in code).
    # openai.api_key = os.getenv("OPENAI_API_KEY", "")

    # Load the input narratives
    with open(args.input_file, 'r', encoding='utf-8') as f:
        narratives = json.load(f)

    results = []
    
    # Load existing results if the output file exists
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

    for narrative in narratives:
        narrative_id = narrative.get('narrative_id', 'unknown_id')
        
        # Skip if this narrative has already been processed and doesn't contain an error
        if any(r['narrative_id'] == narrative_id and 'Error calling model' not in r['model_output'] for r in results):
            print(f"Skipping already processed narrative: {narrative_id}")
            continue

        # Remove any existing error result for this narrative
        results = [r for r in results if r['narrative_id'] != narrative_id]

        synopsis = narrative.get('synopsis', [])

        # Build the prompt
        numbered_story = create_numbered_story(synopsis)
        prompt = (
            f"""Below is a story, broken down into numbered sentences:\n\n
            {numbered_story}\n\n
            
            ### GROUND TRUTH TURNING POINTS
            - Opportunity (tp1): Introductory event that occurs after presenting the setting and background of the main characters.
            - Change of Plans (tp2): Event where the main goal of the story is defined, starting the main action.
            - Point of No Return (tp3): Event that pushes the main character(s) to fully commit to their goal.
            - Major Setback (tp4): Event where things fall apart temporarily or permanently.
            - Climax (tp5): Final event/resolution of the main story (the "biggest spoiler").

            #### Story Arcs:
            -  **Rags to Riches:** Protagonist starts in a disadvantaged situation and ends in a much better one.
            -  **Riches to Rags:** Protagonist starts in a high-status position but ends in a significantly lower state.
            -  **Man in a Hole:** Protagonist falls into a dilemma and finds a way out, ending better than at the beginning.
            -  **Icarus:** Protagonist rises to success but then faces a drastic downfall. 
            -  **Double Man in a Hole:** Protagonist faces two cycles of dilemma and recovery. 
            -  **Cinderella:** Protagonist rises, faces a setback, and ultimately achieves a higher state.
            -  **Oedipus:** Protagonist starts high, falls, recovers, and then faces another significant downfall.
            
            Please identify the sentence indices of any turning points and name the story arc. No need to explain your reasoning. Please format the output as a JSON object in the following format. """,
            "{tp1: ##.#, tp2: ##.#, tp3: ##.#, tp4: ##.#, tp5: ##.#, story_arc: <story_arc_name>}"
        )

        # Call the model (example with OpenAI ChatCompletion)
        # Replace this block with the actual API call for your chosen model or provider.
        try:
            response = client.completions.create(
                model=args.model_name,
                prompt=prompt,
                temperature=0.7
            )
            model_output = response.choices[0].text
        except Exception as e:
            # If there's an error with the API, store the error message instead of a normal response
            model_output = f"Error calling model: {e}"

        # Add result to the collection
        results.append({
            "narrative_id": narrative_id,
            "model_output": model_output
        })

        # Save results after each generation
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved result for narrative: {narrative_id}")

    print(f"Evaluation complete. Results saved to {args.output_file}.")

if __name__ == "__main__":
    main()