import json
import random
import os
from langchain_openai import ChatOpenAI

# Configuration
NUM_NARRATIVES = 5  # Easy to change this value
DATA_DIR = 'data_release'
RESULTS_DIR = 'results'
PROMPTS_DIR = 'prompts'

def load_data():
    """Load all necessary data files."""
    with open(f'{DATA_DIR}/ground_truth_arc.json', 'r') as f:
        ground_truth_arcs = json.load(f)
    
    with open(f'{DATA_DIR}/narratives.json', 'r') as f:
        narratives = json.load(f)
    
    with open(f'{DATA_DIR}/ground_truth_tp.json', 'r') as f:
        ground_truth_tps = json.load(f)
        
    return ground_truth_arcs, narratives, ground_truth_tps

def get_random_human_narratives(narratives, ground_truth_arcs, num_samples=NUM_NARRATIVES):
    """Sample random human-written narratives."""
    # Get human-written narratives
    human_narratives = {id: narrative for id, narrative in narratives.items() 
                       if narrative['source'] == "Human"}
    
    # Keep sampling until we get exactly num_samples human narratives
    random_narratives = {}
    while len(random_narratives) < num_samples:
        num_needed = num_samples - len(random_narratives)
        sample_ids = random.sample(list(ground_truth_arcs.keys()), num_needed)
        
        # Add any that are human narratives
        for id in sample_ids:
            if id in human_narratives:
                random_narratives[id] = human_narratives[id]
    
    return random_narratives

def prepare_selected_narratives(random_narratives, narratives, ground_truth_tps, ground_truth_arcs):
    """Create clean data structure for selected narratives."""
    selected_narratives = {}
    
    for narrative_id in random_narratives:
        narrative = narratives[narrative_id]
        selected_narratives[narrative_id] = {
            'title': narrative['title'],
            'synopsis': narrative['synopsis'],
            'turning_points': ground_truth_tps[narrative_id],
            'arc': ground_truth_arcs[narrative_id]
        }
    
    return selected_narratives

def get_turning_points_from_llm(llm, title, synopsis):
    """Get turning points prediction from LLM."""
    synopsis_prompt = open(f'{PROMPTS_DIR}/base_synopsis.txt').read().format(
        title=title,
        synopsis=synopsis
    )
    context_prompt = open(f'{PROMPTS_DIR}/base_context.txt').read()
    
    messages = [
        {"role": "system", "content": context_prompt},
        {"role": "user", "content": synopsis_prompt}
    ]
    
    response = llm.invoke(messages)
    
    # Extract the JSON part from the response and parse it
    try:
        # The response.content contains the raw text response
        response_text = response.content
        
        # Find the JSON part (assuming it's enclosed in curly braces)
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        json_str = response_text[start_idx:end_idx]
        
        # Parse the JSON string into a Python dictionary
        turning_points = json.loads(json_str)
        return turning_points
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return {"error": "Failed to parse turning points", "raw_response": response_text}

def main(num_narratives=NUM_NARRATIVES):
    # Initialize the LLM
    llm = ChatOpenAI(
        api_key="ollama",
        model="llama3.3",
        base_url="http://localhost:11434/v1",
    )

    # Load all data
    ground_truth_arcs, narratives, ground_truth_tps = load_data()
    
    # Get random narratives
    random_narratives = get_random_human_narratives(
        narratives, 
        ground_truth_arcs, 
        num_samples=num_narratives
    )
    
    # Prepare selected narratives
    selected_narratives = prepare_selected_narratives(
        random_narratives, 
        narratives, 
        ground_truth_tps, 
        ground_truth_arcs
    )

    # Process each narrative through the LLM
    results = {}
    for narrative_id, narrative_data in selected_narratives.items():
        print(f"\nProcessing narrative: {narrative_data['title']}")
        
        llm_response = get_turning_points_from_llm(
            llm,
            narrative_data['title'],
            narrative_data['synopsis']
        )

        
        # Create a results entry with both ground truth and LLM predictions
        results[narrative_id] = {
            'title': narrative_data['title'],
            'synopsis': narrative_data['synopsis'],
            'ground_truth_turning_points': narrative_data['turning_points'],
            'llm_turning_points': llm_response,
            'arc': narrative_data['arc']
        }
        

    # Save results to a file
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f'{RESULTS_DIR}/turning_points_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

