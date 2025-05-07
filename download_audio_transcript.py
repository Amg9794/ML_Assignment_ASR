"""
This script downloads PDF transcripts and MP3 audio files from a given Excel file containing links.
It also converts the MP3 files to WAV format and resamples them to a specified sample rate.
"""


import os
import requests
import pandas as pd
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import subprocess
import librosa
import os
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# Function to download PDF and MP3
def download_files(row, session, pdf_directory, mp3_directory):
    index = row.name
    case_name = row['Case_id']
    pdf_link = row['Transcript Link']
    mp3_link = row.get('mp3 format link')  # use get to avoid key error if column is missing
    hearing_date = row.get('Hearing Date')
    parsed_date = pd.to_datetime(hearing_date).strftime('%Y_%m_%d')
    results = []

    # Download PDF
    pdf_success = False
    if pd.isna(pdf_link):
        results.append((index, False, f"Skipping PDF download for row {index} due to missing PDF link."))
    else:
        pdf_file_name = f"{case_name}_{parsed_date}.pdf"
        pdf_file_path = os.path.join(pdf_directory, pdf_file_name)
        if os.path.exists(pdf_file_path):
            results.append((index, False, f"PDF already exists, skipping: {pdf_file_name}"))
        else:
            try:
                response = session.get(pdf_link, timeout=30)
                response.raise_for_status()
                with open(pdf_file_path, 'wb') as f:
                    f.write(response.content)
                results.append((index, True, f"Downloaded PDF: {pdf_file_name}"))
                pdf_success = True
            except requests.exceptions.RequestException as e:
                results.append((index, False, f"Failed to download PDF {pdf_file_name}: {e}"))

    # Download MP3
    mp3_success = False
    if pd.isna(mp3_link):
        results.append((index, False, f"Skipping MP3 download for row {index} due to missing MP3 link."))
    else:
        # Modify Dropbox link for direct download
        if "dropbox.com" in mp3_link:
            mp3_link = mp3_link.replace("dl=0", "raw=1").replace("dl=1", "raw=1")

        mp3_file_name = f"{case_name}_{parsed_date}.mp3"   # keep same name of mp3 as corresponding pdf
        mp3_file_path = os.path.join(mp3_directory, mp3_file_name)
        if os.path.exists(mp3_file_path):
            results.append((index, False, f"MP3 already exists, skipping: {mp3_file_name}"))
        else:
            try:
                response = session.get(mp3_link, stream=True, timeout=30)
                response.raise_for_status()
                with open(mp3_file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                results.append((index, True, f"Downloaded MP3: {mp3_file_name}"))
                mp3_success = True
            except requests.exceptions.RequestException as e:
                results.append((index, False, f"Failed to download MP3 {mp3_file_name}: {e}"))

    return results, pdf_success, mp3_success

# Function to convert MP3 to WAV using FFmpeg
def convert_mp3_to_wav_with_ffmpeg(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    wav_count = 0
    for audio_file in os.listdir(input_folder):
        if audio_file.endswith(".mp3"):
            input_path = os.path.join(input_folder, audio_file)
            output_path = os.path.join(output_folder, audio_file.replace(".mp3", ".wav"))
            
            if os.path.exists(output_path):
                print(f"WAV already exists, skipping: {output_path}")
                continue 
            try:
                subprocess.run(['ffmpeg', '-i', input_path, output_path], check=True, capture_output=True, text=True)
                print(f"Converted and saved: {output_path}")
                wav_count += 1
            except subprocess.CalledProcessError as e:
                print(f"Error converting {audio_file}: {e.stderr}")
    
    return wav_count

# Function to resample WAV audio
def resample_audio(input_path, sr_out=16000):
    audio_files = [f for f in Path(input_path).rglob("*.wav")]
    resampled_count = 0
    
    for file_path in tqdm(audio_files, desc="Resampling"):
        try:
            # Load audio with original sample rate and preserve channels
            audio, sr_orig = librosa.load(file_path, sr=None, mono=False)
            # Convert to mono if not already
            if audio.ndim > 1:
                audio = librosa.to_mono(audio)
            # Only resample if necessary
            if sr_orig != sr_out:
                audio_resampled = librosa.resample(audio, orig_sr=sr_orig, target_sr=sr_out)
                sf.write(file_path, audio_resampled, sr_out)
                # print(f"Resampled: {file_path} from {sr_orig}Hz to {sr_out}Hz (mono)")
                resampled_count += 1
            else:
                print(f"Skipped: {file_path} (Sample rate already {sr_orig}Hz)")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return resampled_count


def main(base_dir, xls_path, max_workers):
    # Directory to save the downloaded PDFs and MP3s
    pdf_directory = os.path.join(base_dir, 'transcripts_pdfs')
    mp3_directory = os.path.join(base_dir, 'audio_mp3s')
    wav_directory = os.path.join(base_dir, 'audio_wavs')
    os.makedirs(pdf_directory, exist_ok=True)
    os.makedirs(mp3_directory, exist_ok=True)
    os.makedirs(wav_directory, exist_ok=True)
    

    # Load the dataset
    data = pd.read_excel(xls_path)
    data = data.reset_index(drop=True)

    # add'Case_id' column for simplifying the file naming
    if 'Case_id' not in data.columns:
        case_column = [f'case{i+1}' for i in range(len(data))]
        data.insert(1, 'Case_id', case_column)
        # Save the modified DataFrame back to the Excel file
        data.to_excel(xls_path, index=False)
    else:
        # Optionally, verify or update the Case_Num column
        case_column = [f'case{i+1}' for i in range(len(data))]
        if not all(data['Case_id'] == case_column):
            print("Warning: Existing 'Case_id' column does not match expected values. Updating column.")
            data['Case_id'] = case_column
            # Save the updated DataFrame back to the Excel file
            data.to_excel(xls_path, index=False)
    

    # Counters for downloaded files
    pdf_download_count = 0
    mp3_download_count = 0

    # Set up a requests session with retries
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Download PDFs and MP3s in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(download_files, row, session, pdf_directory, mp3_directory): index for index, row in data.iterrows()}
        for future in as_completed(future_to_index):
            results, pdf_success, mp3_success = future.result()
            for index, success, message in results:
                print(message)
            if pdf_success:
                pdf_download_count += 1
            if mp3_success:
                mp3_download_count += 1

    # Close the session
    session.close()

    print("\nConverting MP3s to WAVs...")
    wav_count = convert_mp3_to_wav_with_ffmpeg(mp3_directory, wav_directory)

    # Resample WAV files
    print("\nResampling WAV files...")
    resampled_count = resample_audio(wav_directory)
    
    # Summary of operations
    total_pdfs = len([f for f in os.listdir(pdf_directory) if f.endswith('.pdf')])
    total_mp3s = len([f for f in os.listdir(mp3_directory) if f.endswith('.mp3')])
    total_wavs = len([f for f in os.listdir(wav_directory) if f.endswith('.wav')])
    
    print(f"\nAll files have been processed.")
    print(f"Total PDFs downloaded: {pdf_download_count}")
    print(f"Total MP3s downloaded: {mp3_download_count}")
    print(f"Total PDFs in directory: {total_pdfs}")
    print(f"Total MP3s in directory: {total_mp3s}")
    print(f"Total WAVs in directory: {total_wavs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PDFs and MP3s, convert to WAV, and resample audio.")
    parser.add_argument("--base_dir", type=str, default="/hdd2/Aman/scripts/ML_assign/Dataset/raw", help="Base directory to save the downloaded and processed files.")
    parser.add_argument("--xls_path", type=str, default="/hdd2/Aman/scripts/ML_assign/src/SC_Transcripts_ML_Assignment_Speech.xlsx", help="Path to the Excel file containing the links.")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of worker threads for downloading files.")
    args = parser.parse_args()
    main(args.base_dir, args.xls_path, args.max_workers)
