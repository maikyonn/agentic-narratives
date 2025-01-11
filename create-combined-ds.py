import json
import os

def load_json_file(file_path):
    """
    Loads a JSON file and returns its content.

    :param file_path: Path to the JSON file.
    :return: Parsed JSON data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
            print(f"Successfully loaded {file_path}")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from file {file_path}: {e}")

def combine_narratives(narratives, turning_points, arc_labels):
    """
    Combines narratives with their corresponding turning points and arc labels.

    :param narratives: Dictionary of narratives.
    :param turning_points: Dictionary of turning points.
    :param arc_labels: Dictionary of arc labels.
    :return: List of combined narrative data.
    """
    combined_data = []
    total_narratives = len(narratives)
    processed = 0
    skipped_tp = 0
    skipped_arc = 0

    for narrative_id, narrative_content in narratives.items():
        # Initialize combined entry with narrative ID and all its data
        combined_entry = {
            "narrative_id": narrative_id,
            **narrative_content  # Unpack all key-value pairs from narrative_content
        }

        # Add turning points if available
        if narrative_id in turning_points:
            combined_entry["turning_points"] = turning_points[narrative_id]
        else:
            combined_entry["turning_points"] = None
            skipped_tp += 1
            print(f"Warning: No turning points found for narrative ID: {narrative_id}")

        # Add arc label if available
        if narrative_id in arc_labels:
            combined_entry["arc_label"] = arc_labels[narrative_id]
        else:
            combined_entry["arc_label"] = None
            skipped_arc += 1
            print(f"Warning: No arc label found for narrative ID: {narrative_id}")

        combined_data.append(combined_entry)
        processed += 1
        if processed % 100 == 0 or processed == total_narratives:
            print(f"Processed {processed}/{total_narratives} narratives.")

    print(f"Total narratives processed: {processed}")
    print(f"Narratives missing turning points: {skipped_tp}")
    print(f"Narratives missing arc labels: {skipped_arc}")

    return combined_data

def save_combined_data(combined_data, output_file):
    """
    Saves the combined data to a JSON file.

    :param combined_data: List of combined narrative data.
    :param output_file: Path to the output JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(combined_data, file, indent=4, ensure_ascii=False)
        print(f"Combined data saved to {output_file}")

def main():
    # Define file paths
    narratives_file = 'data_release/narratives.json'
    turning_points_file = 'data_release/ground_truth_tp.json'
    arc_labels_file = 'data_release/ground_truth_arc.json'
    output_file = 'data_release/combined_narratives.json'

    # Load JSON data
    try:
        narratives_data = load_json_file(narratives_file)
        turning_points_data = load_json_file(turning_points_file)
        arc_labels_data = load_json_file(arc_labels_file)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return

    # Combine narratives with turning points and arc labels
    combined_narratives = combine_narratives(narratives_data, turning_points_data, arc_labels_data)

    # Save the combined data to a new JSON file
    save_combined_data(combined_narratives, output_file)

if __name__ == "__main__":
    main()