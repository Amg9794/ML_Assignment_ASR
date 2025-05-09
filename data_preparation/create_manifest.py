import json
import os

# Input and output directories
input_json_folder = "/hdd2/Aman/scripts/ML_assign/spk_dialg_json"
audio_folder = "/hdd2/Aman/scripts/ML_assign/Dataset/audio_wav_wo_sil"
output_manifest_folder = "/hdd2/Aman/scripts/ML_assign/Dataset/manifest"

# Ensure the output folder exists
if not os.path.exists(output_manifest_folder):
    os.makedirs(output_manifest_folder)

# Process all JSON files in the input directory
for filename in os.listdir(input_json_folder):
    if filename.endswith(".json") and not filename.endswith("_manifest.json"):
        # Derive case name and file paths
        case_name = os.path.splitext(filename)[0]  # Remove .json extension
        input_file = os.path.join(input_json_folder, filename)
        output_file = os.path.join(output_manifest_folder, f"{case_name}_manifest.json")
        audio_file = os.path.join(audio_folder, f"{case_name}.wav")
        print(f"Processing {case_name}...")
        try:
            # Read the input JSON file
            with open(input_file, "r") as f:
                input_data = json.load(f)
            # Concatenate all transcripts with '|' separator
            concatenated_text = " | ".join(entry["dialogue"] for entry in input_data)
            # Create the new manifest entry
            manifest_entry = {
                "audio_filepath": audio_file,
                "text": concatenated_text
            }
            # Write to the output JSON manifest file
            with open(output_file, "w") as f:
                json.dump(manifest_entry, f)
                f.write("\n")  # Ensure each line is a separate JSON object
            print(f"Manifest file '{output_file}' has been created.")
        except Exception as e:
            print(f"Error processing {case_name}: {str(e)}")
    else:
        print(f"Skipping file: {filename}")
print("All JSON files processed!")