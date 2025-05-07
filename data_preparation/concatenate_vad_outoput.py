import os
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor

def process_single_file(audio_file, audio_dir, vad_dir, output_dir):
    try:
        audio_path = os.path.join(audio_dir, audio_file)
        vad_path = os.path.join(vad_dir, audio_file.replace('.wav', '.txt'))
        if not os.path.exists(vad_path):
            print(f"VAD file not found for: {audio_file}")
            return
        audio = AudioSegment.from_wav(audio_path)
        output_audio = AudioSegment.silent(duration=0)
        with open(vad_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue

                start_sec = float(parts[0])
                end_sec = float(parts[1])

                start_ms = int(start_sec * 1000)
                end_ms = int(end_sec * 1000)

                segment = audio[start_ms:end_ms]
                output_audio += segment

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, audio_file)
        output_audio.export(output_path, format="wav")
        print(f"Processed: {audio_file}")
    
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")

def process_all_files_parallel(base_dir, output_dir, num_workers=4):
    audio_dir = os.path.join(base_dir, "audio_wavs")
    vad_dir = os.path.join(base_dir, "VAD_seg_txt")

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for audio_file in audio_files:
            futures.append(
                executor.submit(process_single_file, audio_file, audio_dir, vad_dir, output_dir)
            )

        # Optionally wait for all to finish
        for future in futures:
            future.result()

process_all_files_parallel(
    base_dir="/hdd2/Aman/scripts/ML_assign/Dataset/raw",
    output_dir="/hdd2/Aman/scripts/ML_assign/Dataset/audio_wav_wo_sil",
    num_workers=10
)
