'''
This script processes all .wav files in the specified directory, generates VAD-based segment timestamps,
and saves them in a text file with the same name as the audio file.
'''

import os
from pathlib import Path
import argparse
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

# Load the VAD model only once
model = load_silero_vad()
def process_audio(file_path, output_dir):
    try:
        output_file = output_dir / (file_path.stem + '.txt')
        # Skip if the .txt file already exists
        if output_file.exists():
            print(f"Skipping {file_path.name} (already processed)")
            return
        wav = read_audio(str(file_path))
        speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True)
        if not speech_timestamps:
            print(f"No speech found in {file_path.name}")
            return
        with open(output_file, "w") as f:
            for i, seg in enumerate(speech_timestamps, start=1):
                f.write(f"{seg['start']}\t{seg['end']}\tseg{i}\n")
        print(f"Saved segments for {file_path.name}")
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")

def process_all_audio_files(base_dir):
    base_path = Path(base_dir)
    audio_dir = base_path / "audio_wavs"
    output_dir = base_path / "VAD_seg_txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = list(audio_dir.glob("*.wav"))
    if not audio_files:
        print("No .wav files found in audio_wavs/")
        return
    for file_path in audio_files:
        process_audio(file_path, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate VAD-based segment timestamps for .wav files.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/hdd2/Aman/scripts/ML_assign/Dataset/raw",
        help="Base directory that contains 'audio_wavs'."
    )
    args = parser.parse_args()
    process_all_audio_files(args.base_dir)
