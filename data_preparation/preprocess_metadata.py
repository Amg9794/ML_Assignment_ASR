import json
import re
import os  # For working with files and directories

def correct_ordinal_number_spacing(text):
    # Fix "1 st" → "1st"
    text = re.sub(r'(\b\d+)\s+((?:st|nd|rd|th)\b)', r'\1\2', text, flags=re.IGNORECASE)
    # Remove .'
    text = re.sub(r"\.\'", "", text)
    return text

def process_dialogues(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        dialogues = json.load(file)

    changes_made = 0
    for entry in dialogues:
        if "dialogue" in entry:
            original_dialogue = entry["dialogue"]
            processed_dialogue = correct_ordinal_number_spacing(original_dialogue)
            if original_dialogue != processed_dialogue:
                changes_made += 1
                speaker = entry.get('real_speaker', 'Unknown Speaker')
                print(f"\nFile: {json_file_path}")
                print(f"Speaker: {speaker}")
                print(f"Original: {original_dialogue}")
                print(f"Corrected: {processed_dialogue}")
                entry["dialogue"] = processed_dialogue

    print(f"\nTotal dialogues modified in {os.path.basename(json_file_path)}: {changes_made}")
    with open(json_file_path, 'w', encoding='utf-8') as file:
        json.dump(dialogues, file, indent=4, ensure_ascii=False)
    print(f"Changes saved to {json_file_path}\n")

    return dialogues

def process_all_jsons_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            process_dialogues(file_path)

# Usage
if __name__ == "__main__":
    directory_path = "/hdd2/Aman/scripts/ML_assign/spk_dialg_json"  # Your directory with JSONs
    process_all_jsons_in_directory(directory_path)
