import os
import sys
import random
import pandas as pd
import numpy as np

import torch
torch.set_num_threads(8)

current_directory = os.getcwd()
sys.path.append(current_directory)

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from datasets import Audio
import re
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from datasets import DatasetDict,Dataset,concatenate_datasets


chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
def process(save_path):
    train_path = "/hdd2/Aman/scripts/ML_assign/Final_dataset/train.csv"
    dev_path = "/hdd2/Aman/scripts/ML_assign/Final_dataset/validation.csv"

    # def remove_special_characters(batch):
    #     batch["transcription"] = normalizer(batch["transcription"])
    #     return batch
    
    def prepare_dataset(batch):
        audio = batch["path"]
        audio_array = audio['array']
        sampling_rate = audio['sampling_rate']
        batch["input_features"] = feature_extractor(audio_array, sampling_rate=audio["sampling_rate"]).input_features[0]
        # encode target text to label ids 
        batch["labels"] = tokenizer(batch["transcription"]).input_ids
        return batch
    
    audio = []
    transcript= []

    flg=0
    with open(train_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            # print(line)
            if flg==0:        
                flg=1
                continue
            line = line.split(',')
            audio.append(line[0])
            transcript.append(line[1])

    train_df = pd.DataFrame({'path':audio,'transcription':transcript})

    audio = []
    transcript= []

    flg=0
    with open(dev_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            # print(line)
            if flg==0:        
                flg=1
                continue
            line = line.split(',')
            audio.append(line[0])
            transcript.append(line[1])

    dev_df = pd.DataFrame({'path':audio,'transcription':transcript})

    train_data_hf = Dataset.from_pandas(train_df)
    dev_data_hf = Dataset.from_pandas(dev_df)

    dataset = DatasetDict({'train':train_data_hf,'validation':dev_data_hf})

    # normalizer = BasicTextNormalizer()

    # dataset = dataset.map(remove_special_characters)   

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path,cache_dir='/hdd2/Aman/scripts/whisper_hallucination/models')
    tokenizer = WhisperTokenizer.from_pretrained(model_path, language="en", task="transcribe",cache_dir='/hdd2/Aman/scripts/whisper_hallucination/models')
    input_str = dataset["train"][0]["transcription"]
    labels = tokenizer(input_str).input_ids
    decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
    decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

    print(f"Input:                 {input_str}")
    print(f"Decoded w/ special:    {decoded_with_special}")
    print(f"Decoded w/out special: {decoded_str}")
    print(f"Are equal:             {input_str == decoded_str}")

    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

    # dataset['train'] = dataset['train'].select(range(100))
    # dataset['validation'] = dataset['validation'].select(range(100))

    print(dataset)

    # Define cache file paths
    train_cache_path = f"/hdd2/Aman/scripts/whisper_hallucination/models/cache_train.txt"
    val_cache_path = f"/hdd2/Aman/scripts/whisper_hallucination/models/cache_val.txt"

    # Process dataset
    dataset['train'] = dataset['train'].map(
        prepare_dataset, 
        remove_columns=dataset.column_names["train"], 
        num_proc=1,
        cache_file_name=train_cache_path,
        keep_in_memory=False
    )

    dataset['validation'] = dataset['validation'].map(
        prepare_dataset, 
        remove_columns=dataset.column_names["validation"], 
        num_proc=1,
        cache_file_name=val_cache_path,
        keep_in_memory=False
    )

    # Clean up cache files
    if os.path.exists(train_cache_path):
        os.remove(train_cache_path)
    if os.path.exists(val_cache_path):
        os.remove(val_cache_path)
    
    print(dataset)

    dataset.save_to_disk(save_path)

if __name__ == "__main__":
    model_path = "openai/whisper-large-v3"
    process(save_path=f"/hdd2/Aman/preprocess/Whisper_med/SC")

