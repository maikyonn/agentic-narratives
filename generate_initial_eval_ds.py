#!/usr/bin/env python
import os
import json
import argparse
import asyncio
from typing import List, Dict
import aiohttp
from openai import AsyncOpenAI

def create_numbered_story(synopsis_list):
    """
    Convert a list of sentences into a single string,
    each sentence prefixed with its index for clarity.
    """
    return "\n".join(f"{i+1}) {s}" for i, s in enumerate(synopsis_list))

async def process_narrative(client: AsyncOpenAI, narrative: Dict, args: argparse.Namespace) -> Dict:
    """Process a single narrative asynchronously."""
    narrative_id = narrative.get('narrative_id', 'unknown_id')
    synopsis = narrative.get('synopsis', [])
    numbered_story = create_numbered_story(synopsis)
    
    prompt = f"""Below is a story, broken down into numbered sentences:\n\n
            {numbered_story}
            
            ### GROUND TRUTH TURNING POINTS
            - Opportunity (tp1): Introductory event that occurs after presenting the setting and background of the main characters.
            - Change of Plans (tp2): Event where the main goal of the story is defined, starting the main action.
            - Point of No Return (tp3): Event that pushes the main character(s) to fully commit to their goal.
            - Major Setback (tp4): Event where things fall apart temporarily or permanently.
            - Climax (tp5): Final event/resolution of the main story (the "biggest spoiler").

            ### Story Arcs:
            -  **Rags to Riches:** Protagonist starts in a disadvantaged situation and ends in a much better one.
            -  **Riches to Rags:** Protagonist starts in a high-status position but ends in a significantly lower state.
            -  **Man in a Hole:** Protagonist falls into a dilemma and finds a way out, ending better than at the beginning.
            -  **Icarus:** Protagonist rises to success but then faces a drastic downfall. 
            -  **Double Man in a Hole:** Protagonist faces two cycles of dilemma and recovery. 
            -  **Cinderella:** Protagonist rises, faces a setback, and ultimately achieves a higher state.
            -  **Oedipus:** Protagonist starts high, falls, recovers, and then faces another significant downfall.
            
            Please identify the sentence indices of any turning points and name the story arc. Then format the response as: {{tp1: ##.#, tp2: ##.#, tp3: ##.#, tp4: ##.#, tp5: ##.#, story_arc: <arc_name>}}"""
            
    try:
        response = await client.chat.completions.create(
            model=args.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that identifies turning points and story arcs in narratives."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            stream=False
        )
        model_output = response.choices[0].message.content.strip()
    except Exception as e:
        model_output = f"Error calling model: {e}"
        print(f"Error processing narrative {narrative_id}: {e}")

    return {
        "narrative_id": narrative_id,
        "model_output": model_output
    }

async def main():
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
    if args.output_file == "model_evaluation_output.json":
        output_dir = "./step1-landscape-eval"
        os.makedirs(output_dir, exist_ok=True)
        args.output_file = f"{output_dir}/{args.model_name}_evaluation_output.json"

    openai_api_key = "sk-f61b1867d71847c3a65e72c7cacb0fe7"
    openai_api_base = "https://api.deepseek.com"

    client = AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Load the input narratives
    with open(args.input_file, 'r', encoding='utf-8') as f:
        narratives = json.load(f)

    # Load existing results if the output file exists
    results = []
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

    # Filter out already processed narratives
    processed_ids = {r['narrative_id'] for r in results}
    narratives_to_process = [n for n in narratives if n.get('narrative_id', 'unknown_id') not in processed_ids]

    if not narratives_to_process:
        print("All narratives have been processed already.")
        return

    print(f"Processing {len(narratives_to_process)} narratives...")

    # Process narratives concurrently in batches
    batch_size = 50  # Adjust this number based on API rate limits
    for i in range(0, len(narratives_to_process), batch_size):
        batch = narratives_to_process[i:i + batch_size]
        batch_tasks = [process_narrative(client, narrative, args) for narrative in batch]
        
        # Wait for all tasks in the batch to complete
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)

        # Save results after each batch
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results for batch ending at index {i + len(batch)}")

    print(f"Evaluation complete. Results saved to {args.output_file}.")

if __name__ == "__main__":
    asyncio.run(main())