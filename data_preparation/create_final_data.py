import os
import json
import re
import wave
import contextlib
import shutil
import csv
import uuid
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
seg_ctm_dir = '/hdd2/Aman/scripts/ML_assign/Dataset/CTMs/segments'
word_ctm_dir = '/hdd2/Aman/scripts/ML_assign/Dataset/CTMs/words'
json_dir = '/hdd2/Aman/scripts/ML_assign/spk_dialg_json'
audio_dir = '/hdd2/Aman/scripts/ML_assign/Dataset/audio_wav_wo_sil'

output_base = '/hdd2/Aman/scripts/ML_assign/Final_dataset'
output_splits = {'train': 0.7, 'validation': 0.2, 'test': 0.1}
os.makedirs(output_base, exist_ok=True)

split_dirs = {split: os.path.join(output_base, split) for split in output_splits}
for d in split_dirs.values():
    os.makedirs(d, exist_ok=True)

max_segment_duration = 30.0  # seconds
num_threads = 4

# === HELPER FUNCTIONS ===
def sanitize_speaker_name(name):
    name = name.strip()
    name_parts = name.split()
    if not name_parts:
        return "Unknown"
    def is_short(part):
        return len(part) < 3 or part.endswith(".")
    last_name = None
    for part in reversed(name_parts):
        if not is_short(part):
            last_name = part
            break
    if last_name is None:
        for part in name_parts:
            if not is_short(part):
                last_name = part
                break
        else:
            last_name = "Unknown"
    return re.sub(r'[^a-zA-Z0-9]', '_', last_name)

def load_word_ctm(path):
    with open(path, 'r') as f:
        return [(float(p[2]), float(p[2]) + float(p[3]), p[4]) 
                for line in f if (p := line.strip().split()) and len(p) >= 5]

def load_segment_ctm(path):
    segments = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                start = float(parts[2])
                duration = float(parts[3])
                transcript_parts = []
                for p in parts[4:]:
                    if p == "NA": break
                    transcript_parts.append(p)
                transcript = ' '.join(transcript_parts).replace("<space>", " ")
                segments.append((start, start + duration, transcript))
    return segments

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_chunks_within_30s(start, end, words):
    chunks, current_start, idx = [], start, 0
    while current_start < end and idx < len(words):
        current_words = []
        current_end = current_start
        while idx < len(words) and words[idx][0] < end:
            word_start, word_end, word = words[idx]
            if word_start < current_start:
                idx += 1
                continue
            if word_end - current_start > max_segment_duration:
                break
            current_end = word_end
            current_words.append((word_start, word_end, word))
            idx += 1
        if current_words:
            chunks.append((current_start, current_end, " ".join(w[2] for w in current_words)))
            current_start = current_end
        else:
            break
    return chunks

speaker_audio_data = defaultdict(list)

def export_audio_chunks(audio, chunks, speaker, dialogue_number, audio_base_name):
    sanitized_name = sanitize_speaker_name(speaker)
    csv_data = []
    for segment_part, (start, end, transcript) in enumerate(chunks):
        unique_id = uuid.uuid4().hex[:8]
        chunk_audio = audio[start * 1000:end * 1000]
        chunk_audio = chunk_audio.set_channels(1).set_frame_rate(16000)
        filename = f"{audio_base_name}_{dialogue_number}_{segment_part}_{unique_id}.wav"
        speaker_audio_data[sanitized_name].append((filename, transcript, round(end - start, 2), chunk_audio, sanitized_name))
    return

def process_segment(segment, word_segments, metadata, audio, json_index, current_speaker, audio_base_name):
    seg_start, seg_end, seg_text = segment
    while current_speaker < len(metadata) and seg_text not in metadata[current_speaker]['dialogue']:
        current_speaker += 1
    if current_speaker >= len(metadata):
        return
    speaker = metadata[current_speaker]['real_speaker']
    dialogue_number = current_speaker + 1
    matching_words = [w for w in word_segments if w[1] > seg_start and w[0] < seg_end]
    chunks = get_chunks_within_30s(seg_start, seg_end, matching_words)
    export_audio_chunks(audio, chunks, speaker, dialogue_number, audio_base_name)

def process_file(audio_file):
    audio_base_name = os.path.splitext(os.path.basename(audio_file))[0]
    seg_ctm_path = os.path.join(seg_ctm_dir, f"{audio_base_name}.ctm")
    word_ctm_path = os.path.join(word_ctm_dir, f"{audio_base_name}.ctm")
    json_path = os.path.join(json_dir, f"{audio_base_name}.json")
    if not (os.path.exists(seg_ctm_path) and os.path.exists(word_ctm_path) and os.path.exists(json_path)):
        print(f"Skipping {audio_base_name}: Missing required files.")
        return
    word_segments = load_word_ctm(word_ctm_path)
    segment_intervals = load_segment_ctm(seg_ctm_path)
    metadata = load_json(json_path)
    audio = AudioSegment.from_wav(audio_file)
    json_index, current_speaker = 0, 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for segment_index, segment in enumerate(segment_intervals):
            executor.submit(process_segment, segment, word_segments, metadata, audio, json_index, current_speaker, audio_base_name)
    print(f"Processed {audio_base_name}")

def split_and_save():
    csv_files = {split: open(os.path.join(output_base, f"{split}.csv"), 'w', newline='', encoding='utf-8') for split in output_splits}
    writers = {split: csv.writer(f) for split, f in csv_files.items()}
    for w in writers.values():
        w.writerow(["audio_path", "transcription", "length"])

    for speaker, data in speaker_audio_data.items():
        if len(data) < 5:  # avoid splitting tiny speakers
            split = 'train'
            split_dir = split_dirs[split]
            spk_dir = os.path.join(split_dir, speaker)
            os.makedirs(spk_dir, exist_ok=True)
            for filename, transcript, duration, chunk_audio, _ in data:
                path = os.path.join(spk_dir, filename)
                chunk_audio.export(path, format='wav')
                writers[split].writerow([path, transcript, duration])
            continue

        train_val, test = train_test_split(data, test_size=output_splits['test'], random_state=42)
        train, val = train_test_split(train_val, test_size=output_splits['validation'] / (output_splits['train'] + output_splits['validation']), random_state=42)
        split_map = [('train', train), ('validation', val), ('test', test)]
        for split, split_data in split_map:
            split_dir = split_dirs[split]
            spk_dir = os.path.join(split_dir, speaker)
            os.makedirs(spk_dir, exist_ok=True)
            for filename, transcript, duration, chunk_audio, _ in split_data:
                path = os.path.join(spk_dir, filename)
                chunk_audio.export(path, format='wav')
                writers[split].writerow([path, transcript, duration])

    for f in csv_files.values():
        f.close()

def process_all():
    if os.path.exists(output_base):
        shutil.rmtree(output_base)
    os.makedirs(output_base)
    for d in split_dirs.values():
        os.makedirs(d, exist_ok=True)

    for audio_file in os.listdir(audio_dir):
        if audio_file.endswith('.wav'):
            process_file(os.path.join(audio_dir, audio_file))

    split_and_save()
    print("Dataset prepared and split successfully.")

# === RUN ===
process_all()
