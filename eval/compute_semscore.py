
# !pip3 install torch
# !pip3 install transformers
# !pip3 install bert-score

import os
import sys
from collections import Counter, defaultdict
from functools import partial
from itertools import chain
from math import log
from multiprocessing import Pool
from packaging import version
import numpy as np
import traceback
import pandas as pd
from bert_score import score

from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import (AutoModel, AutoTokenizer, BertConfig, GPT2Tokenizer, RobertaTokenizer,
                          RobertaConfig, XLMConfig, XLNetConfig)

from transformers import __version__ as trans_version
import pandas as  pd
import re
import traceback
from transformers import logging
logging.set_verbosity_error()

torch.set_num_threads(4)

import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda:0')
print("Using device:", device)


def get_model(model_type, num_layers, device, all_layers=None):
    model = AutoModel.from_pretrained(model_type, cache_dir='/hdd2/Aman/scripts/SeMaScore/models', device_map=device)
    model.eval()

    if hasattr(model, "decoder") and hasattr(model, "encoder"):
        model = model.encoder

    # drop unused layers
    if not all_layers:
        if hasattr(model, "n_layers"):  # xlm
            assert (
                0 <= num_layers <= model.n_layers
            ), f"Invalid num_layers: num_layers should be between 0 and {model.n_layers} for {model_type}"
            model.n_layers = num_layers
        elif hasattr(model, "layer"):  # xlnet
            assert (
                0 <= num_layers <= len(model.layer)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.layer)} for {model_type}"
            model.layer = torch.nn.ModuleList(
                [layer for layer in model.layer[:num_layers]]
            )
        elif hasattr(model, "encoder"):  # albert
            if hasattr(model.encoder, "albert_layer_groups"):
                assert (
                    0 <= num_layers <= model.encoder.config.num_hidden_layers
                ), f"Invalid num_layers: num_layers should be between 0 and {model.encoder.config.num_hidden_layers} for {model_type}"
                model.encoder.config.num_hidden_layers = num_layers
            elif hasattr(model.encoder, "block"):  # t5
                assert (
                    0 <= num_layers <= len(model.encoder.block)
                ), f"Invalid num_layers: num_layers should be between 0 and {len(model.encoder.block)} for {model_type}"
                model.encoder.block = torch.nn.ModuleList(
                    [layer for layer in model.encoder.block[:num_layers]]
                )
            else:  # bert, roberta
                assert (
                    0 <= num_layers <= len(model.encoder.layer)
                ), f"Invalid num_layers: num_layers should be between 0 and {len(model.encoder.layer)} for {model_type}"
                model.encoder.layer = torch.nn.ModuleList(
                    [layer for layer in model.encoder.layer[:num_layers]]
                )
        elif hasattr(model, "transformer"):  # bert, roberta
            assert (
                0 <= num_layers <= len(model.transformer.layer)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.transformer.layer)} for {model_type}"
            model.transformer.layer = torch.nn.ModuleList(
                [layer for layer in model.transformer.layer[:num_layers]]
            )
        elif hasattr(model, "layers"):  # bart
            assert (
                0 <= num_layers <= len(model.layers)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.layers)} for {model_type}"
            model.layers = torch.nn.ModuleList(
                [layer for layer in model.layers[:num_layers]]
            )
        else:
            raise ValueError("Not supported")
    else:
        if hasattr(model, "output_hidden_states"):
            model.output_hidden_states = True
        elif hasattr(model, "encoder"):
            model.encoder.output_hidden_states = True
        elif hasattr(model, "transformer"):
            model.transformer.output_hidden_states = True
        # else:
        #     raise ValueError(f"Not supported model architecture: {model_type}")

    return model


def get_tokenizer(model_type, use_fast=False):
  tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=use_fast, cache_dir='/hdd2/Aman/scripts/SeMaScore/models')
  return tokenizer

model_type = "bert-base-cased"  
num_layers = 12
tokenizer = get_tokenizer(model_type, use_fast=False)
model = get_model(model_type, num_layers, device)

def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, : lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, : lens[i]] = 1
    return padded, lens, mask

def sent_encode(tokenizer, sent):
    """
    Encoding a sentence based on the tokenizer.
    Tokenizes and adds special tokens for BERT.
    """
    sent = sent.strip()
    if sent == "":
        return tokenizer.build_inputs_with_special_tokens([])  # Empty sentence
    else:
        return tokenizer.encode(
            sent, 
            add_special_tokens=True, 
            max_length=tokenizer.model_max_length, 
            truncation=True
        )


def collate_idf(arr, tokenizer, idf_dict, device):
    """
    Helper function that pads a list of sentences to hvae the same length and
    loads idf score for words in the sentences.

    Args:
        - :param: `arr` (list of str): sentences to process.
        - :param: `tokenize` : a function that takes a string and return list
                  of tokens.
        - :param: `numericalize` : a function that takes a list of tokens and
                  return list of token indexes.
        - :param: `idf_dict` (dict): mapping a word piece index to its
                               inverse document frequency
        - :param: `pad` (str): the padding token.
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """
    # print('Device Collate:', device)
    arr = [sent_encode(tokenizer, a) for a in arr]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]

    pad_token = tokenizer.pad_token_id

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, 0, dtype=torch.float)
    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask

def bert_encode(model, x, attention_mask, all_layers=False):
    # print('Device bert:', device)
    model.eval()
    with torch.no_grad():
        out = model(x, attention_mask=attention_mask, output_hidden_states=all_layers)
    if all_layers:
        emb = torch.stack(out[-1], dim=2)
    else:
        emb = out[0]
    return emb

def get_bert_embedding(
    all_sens,
    model,
    tokenizer,
    idf_dict,
    device,
    batch_size=-1,
    all_layers=False,
):
    """
    Compute BERT embedding in batches.

    Args:
        - :param: `all_sens` (list of str) : sentences to encode.
        - :param: `model` : a BERT model from `pytorch_pretrained_bert`.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `idf_dict` (dict) : mapping a word piece index to its
                               inverse document frequency
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """

    padded_sens, padded_idf, lens, mask = collate_idf(
        all_sens, tokenizer, idf_dict, device=device
    )
    
    if batch_size == -1:
        batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(
                model,
                padded_sens[i : i + batch_size],
                attention_mask=mask[i : i + batch_size],
                all_layers=all_layers,
            )
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=0)

    return total_embedding, mask, padded_idf

idf_dict = defaultdict(lambda: 1.0)
# set idf for [SEP] and [CLS] to 0
idf_dict[tokenizer.sep_token_id] = 0
idf_dict[tokenizer.cls_token_id] = 0


# Function to get the transformed ground truth after applying edit-distance
def printChanges(s1, s2, dp):
    x=[]
    i = len(s1)
    j = len(s2)

   # Check till the end
    while(i > 0 and j > 0):

        # If characters are same
        if s1[i - 1] == s2[j - 1]:
            x.append(s1[i - 1])
            i -= 1
            j -= 1


        # Replace
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            # print("change", s1[i - 1],
                    #   "to", s2[j - 1])
            j -= 1
            i -= 1
            x.append('$')

        # Delete
        elif dp[i][j] == dp[i - 1][j] + 1:
            # print("Delete", s1[i - 1])
            i -= 1
            x.append('-')

        # Add
        elif dp[i][j] == dp[i][j - 1] + 1:
            # print("Add", s2[j - 1])
            j -= 1
            x.append('+')
    while i>0:
        x.append('-')
        # print("Add", s2[i - 1])
        i-=1
    while j>0:
        x.append('+')
        # print("Add", s2[j - 1])
        j-=1
    return x

# Funtion to compute edit-distance
def editDP(s1, s2):

    len1 = len(s1)
    len2 = len(s2)
    dp = [[0 for i in range(len2 + 1)]
             for j in range(len1 + 1)]

    # Initialize by the maximum edits possible
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # Compute the DP Matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):

            # If the characters are same
            # no changes required
            if s2[j - 1] == s1[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]

            # Minimum of three operations possible
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],
                                   dp[i - 1][j - 1],
                                   dp[i - 1][j])

    # Print the steps
    x=printChanges(s1, s2, dp)
    return x

# FUnction to call the edit-distance function (This function compares reference with respect to hypothesis)
def hit_values(ref_li,hyp_li):
    x=[]
    x=editDP(hyp_li,ref_li)
    # print('reference', ref_li)
    # print('hypothesis',hyp_li)
    aligned=x[::-1]
    aligned=''.join(map(str, aligned))
    # print('aligned:',aligned)
    return aligned

# Funtions to map the ground truth and hypothesis with the transformed ground truth
def sub_sentence_mapper1(ground_truth,aligned):
    split_ground_truth_v1=[]
    split_aligned_v1=[]
    count=0
    index=0
    temp_str1,temp_str2='',''
    while count<len(ground_truth) and index<len(aligned):
        if (ground_truth[count]==' ' and aligned[index]==' '):# or (ground_truth[count]==' ' and aligned[index]=='+'):
            split_ground_truth_v1.append(temp_str1)
            split_aligned_v1.append(temp_str2)
            temp_str1=''
            temp_str2=''
            count+=1
            index+=1
        else:
            if aligned[index]=='-':
                temp_str2+=aligned[index]
                index+=1
            else:
                temp_str1+=ground_truth[count]
                temp_str2+=aligned[index]
                count+=1
                index+=1

    while count<len(ground_truth):
        temp_str1+=ground_truth[count]
        count+=1

    while index<len(aligned):
        temp_str2+=aligned[index]
        index+=1
    split_ground_truth_v1.append(temp_str1)
    split_aligned_v1.append(temp_str2)
    # print(f'#{split_ground_truth_v1}')
    # print(f'#{split_aligned_v1}')
    return split_ground_truth_v1, len(split_ground_truth_v1)==len(split_aligned_v1)

def sub_sentence_mapper2(ground_truth,aligned):
    split_ground_truth_v1=[]
    split_aligned_v1=[]
    count=0
    index=0
    temp_str1,temp_str2='',''
    while count<len(ground_truth) and index<len(aligned):
        if (ground_truth[count]==' ' and aligned[index]==' '):# or (ground_truth[count]==' ' and aligned[index]=='+'):
            split_ground_truth_v1.append(temp_str1)
            split_aligned_v1.append(temp_str2)
            temp_str1=''
            temp_str2=''
            count+=1
            index+=1
        else:
            if aligned[index]=='+':
                temp_str2+=aligned[index]
                index+=1
            else:
                temp_str1+=ground_truth[count]
                temp_str2+=aligned[index]
                count+=1
                index+=1

    while count<len(ground_truth):
        temp_str1+=ground_truth[count]
        count+=1

    while index<len(aligned):
        temp_str2+=aligned[index]
        index+=1
    split_ground_truth_v1.append(temp_str1)
    split_aligned_v1.append(temp_str2)
    # print(f'#{split_ground_truth_v1}')
    # print(f'#{split_aligned_v1}')
    # print(len(split_ground_truth_v1)==len(split_aligned_v1))
    return split_ground_truth_v1, len(split_ground_truth_v1)==len(split_aligned_v1)

# Funtion to compute (1 - match error rate)
def get_mer(ground_truth,inference):
    aligned=hit_values(ground_truth,inference)
    mismatches=0
    for i in aligned:
        if i in ('+','-','$'):
            mismatches+=1
    mer=mismatches/max(len(ground_truth),len(inference))
    # print('Mer:',mer)
    # print('1-Mer:',1- mer)
    return 1-mer

# Function to get the aligned ground truth and aligned hypothesis along with (1 - match error rate)
def mapped_sentence(ground_truth,inference):
    aligned=hit_values(ground_truth,inference)
    mismatches=0
    for i in aligned:
        if i in ('+','-','$'):
            mismatches+=1
    mer=mismatches/max(len(ground_truth),len(inference))
    # split_ground_truth=ground_truth.split(' ')
    split_ground_truth_v1=[]
    split_inference=inference.split(' ')
    split_aligned_v1=[]
    count=0
    index=0
    temp_str1,temp_str2='',''

    mapped_ground_truth,mapped_ground_truth_res=sub_sentence_mapper1(ground_truth,aligned)
    mapped_inference,mapped_inference_res=sub_sentence_mapper2(inference,aligned)
    if mapped_ground_truth_res and mapped_inference_res:
        return mapped_ground_truth,mapped_inference,aligned,mer
    else:
        return False, False, False, False

# Function to compute cosine similarity
def cos_sim(a,b):
    a=a.unsqueeze(0)
    b=b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    # print(a_norm.shape)
    # print(b_norm.shape)
    ss  =  torch.mm(a_norm, b_norm.transpose(0, 1)).item()
    #   print(f'SS:{ss}')
    return ss


def get_gt_embeddings_v1(ground_truth_v1,gt_embedding,gt_tokens):
    start, end, k = 0, 0, 0
    gt_embedding_v1 = []
    test_hyp = []
    
    try:
        for j in range(len(ground_truth_v1)):
            # print(f"\nProcessing aligned word {j}: {ground_truth_v1[j]}")
            # print(f"Current k: {k}, len(gt_tokens): {len(gt_tokens)}")
            
            if k >= len(gt_tokens):
                print(f"Warning: k ({k}) exceeds token length ({len(gt_tokens)})")
                break
                
            start = k+1
            built_word = []  # Store tokens in a list to control spacing
            
            while k < len(gt_tokens):
                current_token = gt_tokens[k].replace('##', '')
                # print(f"Processing token: {gt_tokens[k]} -> {current_token}")
                
                # If it's a continuation token (##), append to last token
                if gt_tokens[k].startswith('##') and built_word:
                    built_word[-1] = built_word[-1] + current_token
                else:
                    built_word.append(current_token)
                
                # Join tokens with space and compare
                current_word = ' '.join(built_word)
                # print(f'k={k}, start={start}, built_tokens={built_word}, current_word="{current_word}", target="{ground_truth_v1[j]}"')
                
                if current_word == ground_truth_v1[j].strip():
                    test_hyp.append(current_word)
                    
                    # Check embedding indices
                    if start <= k+1 and k + 2 <= gt_embedding[0].shape[0]:
                        embedding = torch.mean(gt_embedding[0][start:k+2], dim=0)
                        gt_embedding_v1.append(embedding)
                        # print(f"Successfully added embedding for word: {current_word}")
                        k += 1
                        break
                    else:
                        # print(f"Invalid embedding indices: start={start}, k={k}, embedding_shape={gt_embedding[0].shape}")
                        raise IndexError(f"Invalid embedding indices: start={start}, k={k}, embedding_shape={gt_embedding[0].shape}")
                
                k += 1
                if k >= len(gt_tokens):
                    print(f"Reached end of tokens while processing: {current_word}")
                    break
            
            if k > start and ' '.join(built_word) != ground_truth_v1[j].strip():
                print(f"Warning: Could not match word {ground_truth_v1[j]}")
                # print(f"Final built tokens: {built_word}")
                # print(f"Final word attempt: {' '.join(built_word)}")
        
        # print("\nFinal Results_gt:")
        # print("Matched words:", test_hyp)
        # print(f"Embeddings length: {len(gt_embedding_v1)}, Aligned length: {len(ground_truth_v1)}")
        
        return gt_embedding_v1
    
    except Exception as e:
        print(f"\nError occurred at:")
        print(f"j={j}, k={k}, start={start}")
        print(f"Current word being processed: {' '.join(built_word)}")
        print(f"Aligned word: {ground_truth_v1[j]}")
        print(f"Token list length: {len(gt_tokens)}")
        print(f"Embedding shape: {ground_truth_v1[0].shape}")
        raise e


def get_hyp_embeddings_v1(aligned_v1, hyp_embedding, hyp_tokens):
    start, end, k = 0, 0, 0
    hyp_embeddings_v1 = []
    test_hyp = []
    
    try:
        for j in range(len(aligned_v1)):
            # print(f"\nProcessing aligned word {j}: {aligned_v1[j]}")
            # print(f"Current k: {k}, len(hyp_tokens): {len(hyp_tokens)}")
            
            if k >= len(hyp_tokens):
                # print(f"Warning: k ({k}) exceeds token length ({len(hyp_tokens)})")
                break
                
            start = k+1
            built_word = []  # Store tokens in a list to control spacing
            
            while k < len(hyp_tokens):
                current_token = hyp_tokens[k].replace('##', '')
                # print(f"Processing token: {hyp_tokens[k]} -> {current_token}")
                
                # If it's a continuation token (##), append to last token
                if hyp_tokens[k].startswith('##') and built_word:
                    built_word[-1] = built_word[-1] + current_token
                else:
                    built_word.append(current_token)
                
                # Join tokens with space and compare
                current_word = ' '.join(built_word)
                # print(f'k={k}, start={start}, built_tokens={built_word}, current_word="{current_word}", target="{aligned_v1[j]}"')
                
                if current_word == aligned_v1[j].strip():
                    test_hyp.append(current_word)
                    
                    # Check embedding indices
                    if start <= k+1 and k + 2 <= hyp_embedding[0].shape[0]:
                        embedding = torch.mean(hyp_embedding[0][start:k+2], dim=0)
                        hyp_embeddings_v1.append(embedding)
                        # print(f"Successfully added embedding for word: {current_word}")
                        k += 1
                        break
                    else:
                        # print(f"Invalid embedding indices: start={start}, k={k}, embedding_shape={hyp_embedding[0].shape}")
                        raise IndexError(f"Invalid embedding indices: start={start}, k={k}, embedding_shape={hyp_embedding[0].shape}")
                
                k += 1
                if k >= len(hyp_tokens):
                    # print(f"Reached end of tokens while processing: {current_word}")
                    break
            
            if k > start and ' '.join(built_word) != aligned_v1[j].strip():
                print(f"Warning: Could not match word {aligned_v1[j]}")
                # print(f"Final built tokens: {built_word}")
                # print(f"Final word attempt: {' '.join(built_word)}")
        
        # print("\nFinal Results_hyp:")
        # print("Matched words:", test_hyp)
        # print(f"Embeddings length: {len(hyp_embeddings_v1)}, Aligned length: {len(aligned_v1)}")
        
        return hyp_embeddings_v1
    
    except Exception as e:
        print(f"\nError occurred at:")
        print(f"j={j}, k={k}, start={start}")
        print(f"Current word being processed: {' '.join(built_word)}")
        print(f"Aligned word: {aligned_v1[j]}")
        print(f"Token list length: {len(hyp_tokens)}")
        print(f"Embedding shape: {hyp_embedding[0].shape}")
        raise e

# Function to get generate semascore
def generate_sema_score(ground_truth,hypothesis):

    gt_embedding, masks, padded_idf = get_bert_embedding([ground_truth], model, tokenizer, idf_dict, device=device, all_layers=False)
    hyp_embedding, masks, padded_idf = get_bert_embedding([hypothesis], model, tokenizer, idf_dict, device=device, all_layers=False)
    ground_truth_v1,aligned_v1,aligned,mer=mapped_sentence(ground_truth,hypothesis)
    gt_tokens = [tokenizer.decode([i]) for i in sent_encode(tokenizer, ground_truth)][1:-1]
    hyp_tokens = [tokenizer.decode([i]) for i in sent_encode(tokenizer, hypothesis)][1:-1]



    gt_embeddings_v1=get_gt_embeddings_v1(ground_truth_v1,gt_embedding,gt_tokens)
    hyp_embeddings_v1=get_hyp_embeddings_v1(aligned_v1,hyp_embedding,hyp_tokens)
    total_gt_embedding=torch.mean(gt_embedding[0][1:-1],dim=0)
    
    ss_list=[]
    importance_list=[]
    multiplication_list=[]
    mer_list=[]
    average=0
    metric=0
    # print('gt_embedding len', len(gt_embeddings_v1))
    for j in range(len(gt_embeddings_v1)):
        mer_word=0
        importance = cos_sim(gt_embeddings_v1[j], total_gt_embedding)
        ss = cos_sim(gt_embeddings_v1[j], hyp_embeddings_v1[j])
        importance=(importance+1)/2
        ss=(ss+1)/2
        mer_word=get_mer(ground_truth_v1[j],aligned_v1[j])
        mer_list.append(mer_word)
        metric+=importance*ss*mer_word
        average+=importance
        ss_list.append(round(ss,4))
        importance_list.append(round(importance,4))
        multiplication_list.append(round(importance*ss,4))
        #   print(f'{round(ss,4)}#{ground_truth_v1[j]}#{aligned_v1[j]}')
        #   print(f'{round(importance,4)}#{ground_truth_v1[j]}#{ground_truth}')
    metric/=average
    #   print(metric)
    return (metric,ground_truth_v1,aligned_v1,aligned,ss_list,importance_list,multiplication_list,mer_list)

# Function to generate BERTScore
from bert_score import score
def generate_bert_score_v1(ground_truth, hypothesis, device, lang):
    (P, R, F) = score(
        [ground_truth], 
        [hypothesis], 
        lang=lang, 
        model_type='bert-base-uncased', 
        idf=False, 
        device=device
    )
    # print(f"P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}")
    bert_score_v1 = F.mean().item()
    return bert_score_v1


# Define the main function to generate scores
def generate_Scores(path, lang, lang_acronym, summary_data, device):
    df = pd.read_csv(path)
    exception_data = {}
    bert_scores = []
    sema_scores = []
    wer_scores = []

    # Initialize progress bar
    with tqdm(total=len(df), desc=f"Processing {lang}", unit="row") as pbar:
        for i in range(len(df)):
            try:
                new_df = df.iloc[[i]].copy()
                ground_truth = df.loc[i, 'reference'].strip()
                hypothesis = df.loc[i, 'hypothesis'].strip()

                # Generate BERTScore and SeMaScore
                bert_score_v1 = generate_bert_score_v1(ground_truth, hypothesis, device, lang_acronym)
                sema_score = generate_sema_score(ground_truth, hypothesis)  # Assuming this function is defined elsewhere
            
                # Store scores for averaging
                bert_scores.append(bert_score_v1)
                sema_scores.append(sema_score[0])

                # Add the scores and results to the new DataFrame
                new_df['bert_score'] = bert_score_v1
                new_df['sema_score'] = sema_score[0]
  

                # Ensure 'id' from input CSV is included in the output
                new_df['id'] = df.loc[i, 'id']
                # Reorder columns to ensure 'id' is the first column
                column_order = ['id'] + [col for col in new_df.columns if col != 'id']
                new_df = new_df[column_order]

                # Define the path for saving results
                saving_path = path[:path.rindex('/')] + f'/{lang}_Wm_PT_semascore.csv'
                new_df.to_csv(saving_path, mode='a', index=False, header=(not os.path.exists(saving_path)))

            except Exception as e:
                traceback.print_tb(e.__traceback__)
                print(f'Exception occurred for row {i} in language: {lang}')
                exception_data['ground_truth'] = [ground_truth]
                exception_df = pd.DataFrame(exception_data)

                # Define the path for saving exceptions
                saving_path = path[:path.rindex('/')] + f'/{lang}_Wm_PT_exceptions.csv'
                exception_df.to_csv(saving_path, mode='a', index=False, header=(not os.path.exists(saving_path)))
                new_df.to_csv(saving_path, mode='a', index=False, header=(not os.path.exists(saving_path)))

            # Update progress bar
            pbar.update(1)

    # Calculate averages
    if bert_scores and sema_scores:
        avg_bert = sum(bert_scores) / len(bert_scores)
        avg_sema = sum(sema_scores) / len(sema_scores)

        # Append averages to summary data
        summary_data.append({
            "Language": lang,
            "NumSamples": len(bert_scores),
            "AverageBERTScore": avg_bert,
            "AverageSEMAScore": avg_sema,
        })

        # Print averages
        print("\nFinal Averages for:", lang)
        print(f"Average BERT Score: {avg_bert:.4f}")
        print(f"Average SEMA Score: {avg_sema:.4f}")

    return summary_data



# Save summary file
def save_summary_file(summary_data, summary_path):
    with open(summary_path, 'w') as f:
        f.write("Language Averages Summary\n")
        f.write("=========================\n\n")
        for data in summary_data:
            f.write(f"Language: {data['Language']}\n")
            f.write(f"Number of Samples Processed: {data['NumSamples']}\n")
            f.write(f"Average BERT Score: {data['AverageBERTScore']:.4f}\n")
            f.write(f"Average SEMA Score: {data['AverageSEMAScore']:.4f}\n")
            f.write("\n")


# Language acronyms mapping
language_acronyms = {
    "english": "en"
}

languages = ["english"]
summary_data = []

for lang in languages:
    path = f'/hdd2/Aman/scripts/ML_assign/results/{lang}_new_test_PT_WL.csv'
    lang_acronym = language_acronyms[lang]  # Get the acronym for the current language
    summary_data = generate_Scores(path, lang, lang_acronym, summary_data,device=device)

# Save the summary file
summary_path = "/hdd2/Aman/scripts/ML_assign/results/Average_PT_WL.txt"
