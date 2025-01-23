#!/usr/bin/env python
import os
import json
import argparse
import asyncio
from typing import List, Dict
from openai import AsyncOpenAI

def create_numbered_story(synopsis_list):
    """
    Convert a list of sentences into a single string,
    each sentence prefixed with its index for clarity.
    """
    return "\n".join(f"{i+1}) {s}" for i, s in enumerate(synopsis_list))

def get_previous_sentences(synopsis: List[str], current_index: int) -> str:
    """
    Retrieve all sentences before the current index as a single string.
    """
    return " ".join(synopsis[:current_index])

async def process_sentence(client: AsyncOpenAI, protagonist: str, previous_text: str, current_sentence: str, args: argparse.Namespace) -> Dict:
    """
    Process a single sentence to get three emotional adjectives for the protagonist.
    """
    prompt = f"""The main character of this story is {protagonist}.
    
    Here is the story so far:
    {previous_text}

    Current sentence:
    {current_sentence}
    
    ### Task
    Infer exactly three adjectives describing the protagonist's emotions in the current sentence.
    Format your response as a JSON array containing exactly three adjectives, like this:
    ["adjective1", "adjective2", "adjective3"]"""
    
    try:
        response = await client.chat.completions.create(
            model=args.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that identifies emotional adjectives in narratives."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            stream=False
        )
        adjectives = response.choices[0].message.content.strip()
        return {"adjectives": adjectives}
    except Exception as e:
        print(f"Error processing sentence: {e}")
        return {"adjectives": []}

async def process_narrative(client: AsyncOpenAI, narrative: Dict, args: argparse.Namespace) -> Dict:
    """
    Process a single narrative by calculating arousal scores for each sentence.
    """
    narrative_id = narrative.get('narrative_id', 'unknown_id')
    synopsis = narrative.get('synopsis', [])
    
    # Identify the protagonist of the story (simple assumption for now)
    protagonist_prompt = f"""Here is a story:\n{create_numbered_story(synopsis)}\n\nWho is the main character of this story?"""
    try:
        protagonist_response = await client.chat.completions.create(
            model=args.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that identifies the main character of a story. Simply print out the name of the main character in quotes. Ex: The protagonist is 'John Park'"},
                {"role": "user", "content": protagonist_prompt}
            ],
            temperature=0.7,
            stream=False
        )
        protagonist = protagonist_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error identifying protagonist for narrative {narrative_id}: {e}")
        protagonist = "the protagonist"

    arousal_scores = []
    for i, sentence in enumerate(synopsis):
        previous_text = get_previous_sentences(synopsis, i)
        score = await process_sentence(client, protagonist, previous_text, sentence, args)
        arousal_scores.append(score)

    return {
        "narrative_id": narrative_id,
        "arousal_scores": arousal_scores
    }

async def main():
    parser = argparse.ArgumentParser(
        description="Script to query a model for arousal score calculation, sentence by sentence."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-chat",
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
        default="arousal_scores_output.json",
        help="Path to the output JSON file."
    )
    args = parser.parse_args()

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
    batch_size = 100  # Lower batch size due to sequential sentence processing
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