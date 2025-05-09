import os
import subprocess
import time
import torch

languages = ["English"]

# Common parameters
model_path = "openai/whisper-large-v3" "
batch_size = 4
beam_search = 1
# do_normalize = False

data_dir_base = "/hdd2/Aman/scripts/ML_assign/Final_dataset"  
# bucket_csv_base = "dataset/kathbath/kb_data_clean_wav"
save_path_base = "/hdd2/Aman/scripts/ML_assign/results"
log_file = "/hdd2/Aman/scripts/ML_assign/results/new_test_PT_WL.log"
wer_save_path = "/hdd2/Aman/scripts/ML_assign/results/new_test_WER_PT_WL.txt"

# Create directories if they don't exist
os.makedirs(os.path.dirname(log_file), exist_ok=True)
os.makedirs(save_path_base, exist_ok=True)

# prompt = False 
# apply_lora = False 

# Initialize log file
with open(log_file,'w') as log:
    log.write(f"Script started at {os.popen('date').read()}")
    log.write(f"model_path_used= {model_path}")
    log.write(f"wer_save_path= {wer_save_path}")


# Iterate over each language and run the Python script
for lang in languages:
    data_dir = f"{data_dir_base}/test"
    # bucket_csv = f"{data_dir_base}/{lang}/test/bucket.csv"
    bucket_csv = "/hdd2/Aman/scripts/ML_assign/Final_dataset/test.csv"
    save_path = f"{save_path_base}/{lang}_new_test_PT_WL.csv"

    print(f"Running for language: {lang}")
    start_time = time.time()
    
    try:
        subprocess.run([
            "python", "whisper_inference.py",
            "--model_path", model_path,
            "--batch_size", str(batch_size),
            # "--do_normalize", do_normalize,
            "--language", lang,
            "--data_dir", data_dir,
            "--bucket_csv", bucket_csv,
            # "--chunk_size", str(chunk_size),
            "--save_path", save_path,
            "--wer_save_path", wer_save_path,
        ], check=True)
        
        end_time = time.time()
        elapsed_time = (end_time - start_time)/60

        with open(log_file, 'a') as log:
            log.write(f"Completed run at {os.popen('date').read()}")
            log.write(f"Successfully completed for language: {lang}\n")
            log.write(f"Time taken for {lang}: {elapsed_time:.2f} minutes\n")

    except subprocess.CalledProcessError as e:
        # end_time = time.time()
        # elapsed_time = end_time - start_time

        with open(log_file, 'a') as log:
            log.write(f"Error running for language: {lang}\n")
            # log.write(f"Time taken before error for {lang}: {elapsed_time:.2f} seconds\n")


with open(log_file, 'a') as log:
    log.write(f"Completed run at {os.popen('date').read()}")