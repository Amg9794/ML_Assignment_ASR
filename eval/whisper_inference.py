import os
import sys
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import datasets
import torch
import numpy as np
from datasets import Dataset
import wandb
from peft import PeftModel, PeftConfig

# import torch 
# torch.set_num_threads(6)

os.environ['WANDB_DISABLED'] = 'true'

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor
)
import pandas as pd
from tqdm import tqdm

import jiwer
from whisper.normalizers import IndicTextNormalizer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for model training")
    parser.add_argument("--model_path", type=str, default="openai/whisper-medium", help="Path to the trained model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--do_normalize", action='store_true',default = False , help="Whether to normalize the input data")
    parser.add_argument("--language", type=str, default='hi', help="Language of the dataset")
    parser.add_argument("--data_dir", type=str, default="/hdd2/Aman/scripts/ML_assign/Final_dataset", help="Directory containing the dataset")
    parser.add_argument("--bucket_csv", type=str, default='/hdd2/Aman/scripts/ML_assign/Final_dataset/test.csv', help="CSV file containing bucket information")
    # parser.add_argument("--chunk_size", type=int, default=64, help="Size of data chunks")
    parser.add_argument("--save_path", type=str, default='/hdd2/Aman/scripts/ML_assign/results/ENg_wh_medium_PT.csv', help="Path to save the results")
    parser.add_argument("--wer_save_path", type=str, default='Results/wer.txt', help="Path to save the results")
    parser.add_argument("--prompt", type=bool, default=False, help="use prompt?")
    parser.add_argument("--apply_lora", type=bool, default=False, help="used lora?")
    parser.add_argument("--beam_search", type=int, default=1, help="beam search?")

    return parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["WANDB_MODE"]= "disabled"

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: Any
    decoder_start_token_id: int
    forward_attention_mask: bool
    audio_column_name: str
    do_normalize: bool

    def __call__(
        self, features
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        
        model_input_name = self.processor.model_input_names[0]
        
        features = [
            prepare_dataset(
                feature, 
                audio_column_name=self.audio_column_name, 
                model_input_name=model_input_name,
                feature_extractor=self.processor.feature_extractor,
                do_normalize=self.do_normalize
            ) for feature in features
        ]
        
        input_features = [
            {model_input_name: feature[model_input_name]} for feature in features
        ]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        return batch


def prepare_dataset(batch, audio_column_name, model_input_name, feature_extractor, do_normalize):
    # process audio
    sample = batch[audio_column_name]
    sampling_rate=sample["sampling_rate"]
    inputs = feature_extractor(
        sample["array"],
        sampling_rate=sample["sampling_rate"],
        do_normalize=do_normalize,
    )
    # process audio length
    batch[model_input_name] = inputs.get(model_input_name)[0]
    return batch

@dataclass
class Config:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    audio_column_name: str = field(
        default="audio",
        metadata={
            "help": "The name of the dataset column containing the audio data. Defaults to 'audio'"
        },
    )
    num_workers: int = field(
        default=2,
        metadata={
            "help": "The number of workers for preprocessing"
        },
    )
    use_bettertransformer: bool = field(default=False, metadata={
            "help": "Use BetterTransformer (https://huggingface.co/docs/optimum/bettertransformer/overview)"
        })
    do_normalize: bool = field(default=False, metadata={
            "help": "Normalize in the feature extractor"
        })

def get_prompt_ids(text: str, return_tensors="np"):
    """Converts prompt text to IDs that can be passed to [`~WhisperForConditionalGeneration.generate`]."""
    batch_encoding = tokenizer("<|startofprev|>", " " + text.strip(), add_special_tokens=False)
    batch_encoding.convert_to_tensors(tensor_type=return_tensors)
    return batch_encoding["input_ids"]

if __name__ == "__main__":

    CFG = parse_args()

    # print(f"Model Path: {CFG.model_path}")
    # print(f"Batch Size: {CFG.batch_size}")
    # print(f"Do Normalize: {CFG.do_normalize}")
    print(f"Language: {CFG.language}")
    print(f"Bucket CSV: {CFG.bucket_csv}")


    cfg = Config(
        model_name_or_path=CFG.model_path,
        audio_column_name="audio",
        num_workers=2,
        do_normalize=False,
    )

    training_args = Seq2SeqTrainingArguments(
        # Define your training arguments here
        output_dir="./",
        predict_with_generate = True,
        remove_unused_columns=False,
        disable_tqdm=True,
        report_to = None,
        # use_cpu=True,
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        cfg.model_name_or_path,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        cfg.model_name_or_path,cache_dir = '/hdd2/Aman/scripts/whisper_hallucination/models'
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        cfg.model_name_or_path,
        config=config,
        cache_dir = '/hdd2/Aman/scripts/whisper_hallucination/models',
        # device_map='cpu'
    )

    # model = model.to('cpu')

    tokenizer = WhisperTokenizer.from_pretrained(cfg.model_name_or_path,cache_dir = '/hdd2/Aman/scripts/whisper_hallucination/models',language= "English")

    processor = WhisperProcessor.from_pretrained(cfg.model_name_or_path,cache_dir = '/hdd2/Aman/scripts/whisper_hallucination/models')

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    print("language :  ->>>> ",CFG.language)

    model.generation_config.language = CFG.language
    # model.generation_config.no_repeat_ngram_size = 3    

    if CFG.beam_search!=1:
        print("Beam search applied ", CFG.beam_search)
        model.generation_config.num_beams= CFG.beam_search

    if CFG.apply_lora:
        peft_config = PeftConfig.from_pretrained(cfg.model_name_or_path)
        model = PeftModel.from_pretrained(model, cfg.model_name_or_path)
        print("Lora added")
        
    df = pd.read_csv(CFG.bucket_csv)
    test_files = df['audio_path'].tolist() 
        
    # audio_files = list(map(str, Path(data_dir).glob("*.mp3")))
    audio_files = list(map(str, Path(CFG.data_dir).rglob("*.wav")))
    audio_files = [file for file in audio_files if file in test_files]
  

    ds = Dataset.from_dict({"audio": audio_files})
    ds = ds.map(lambda x: {"id": Path(x["audio"]).stem, "filesize": os.path.getsize(x["audio"])}, num_proc=cfg.num_workers)
    ds = ds.cast_column(
        cfg.audio_column_name,
        datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
    )

    # sort by filesize to minimize padding
    ds = ds.sort("filesize")
    ds = ds.add_column("idx", range(len(ds)))
    # save ids
    ds.remove_columns([x for x in ds.column_names if x != "id"]).to_json("ids.json")
    # ds = ds.select(range(64))

    # df = pd.read_csv(CFG.bucket_csv)
    # test_files = df['file_path'].tolist() 

    mapping = {}
    for i,row in df.iterrows():
        id = row['audio_path'].split('/')[-1][:-4]
        # print(id)
        mapping[id] = row['transcription']

    ground_truth = []

    for id in ds['id']:
        ground_truth.append(mapping[id])

    ds = ds.add_column('ground_truth',ground_truth)

    model_input_name = feature_extractor.model_input_names[0]

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=False,
        audio_column_name=cfg.audio_column_name,
        do_normalize=cfg.do_normalize,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=feature_extractor,
        data_collator=data_collator,       
    )


    text_preds = []

    for num, i in enumerate(tqdm(range(0, len(ds), CFG.batch_size), desc="Inference", unit="chunk")):
        ii = min(i+CFG.batch_size, len(ds))
        temp = ds.select(range(i, ii))

        predictions = trainer.predict(temp).predictions
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        text_preds.extend(predictions)

    data = pd.DataFrame({'id':ds['id'],'hypothesis':text_preds,'reference':ds['ground_truth']})
    normalizer = BasicTextNormalizer()
    data.to_csv(CFG.save_path)

    data["hypothesis"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference"] = [normalizer(text) for text in data["reference"]]

    data.to_csv(CFG.save_path)

    wer = jiwer.wer(list(data["reference"]), list(data["hypothesis"]))

    print()
    print(f"{CFG.language} WER: {wer * 100:.2f} %")
    with open(CFG.wer_save_path,'a+') as f:
        f.write(f"{CFG.language} WER: {wer * 100:.2f} % \n")
    print()

    data.to_csv(CFG.save_path)

