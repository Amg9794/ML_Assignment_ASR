# ML_Assignment_ASR

# **🎯Objective**

The aim of this project was to create and process a raw dataset suitable for finetuning ASR model for transcribing Supreme Court hearings. 

# **📝Approach Overview**

I followed a structured approach to solve the problem, including following steps :

1- Data collection : Involves downloading raw data.

2- Data preprocessing :- Removing silence

3- Data Preparation :- It involves getting speaker and their corresponds dialogue alignment and making it standard form to FT ASR.

4-  Model selection : Tried whisper variants

5- Evaluation : Based on WER, CER and SEMA SCORE

# **🎵🗒️ Data Collection**

 Before downloading the transcripts and corresponding court hearing from links given in excel sheet. 

I found in Excel sheet some of cases name were NAN and some case names were common so to simplify the naming and proceed further I’ve added one column named as **Case ID** which assign case id (like case1, case2…and so on)  to each entries in excel sheet as uses case id along with hearing date as identifier for each case. (ex- case1_2024_02_12).

To download raw data i used this script [download_audio_trancript.py](https://github.com/Amg9794/ML_Assignment_ASR/blob/main/data_preparation/download_audio_transcript.py). 

This script download audio in mp3 format and their corresponding transcript pdf and convert mp3 audio to mono channel 16KHz audio in .wav format. 

***Why We Chose WAV Format Over Others*** : WAV is an industry-standard format widely used in professional settings. As an uncompressed format, it preserves all audio details, making it ideal for Speech-to-Text applications, particularly in fields requiring precise transcription like legal or medical work.

(However, this approach has a limitation: converting from a lossy format (MP3) to a lossless format (WAV) does not restore any previously lost data. The audio quality remains identical to the original MP3 file. The WAV conversion serves only to ensure compatibility with ASR models like Whisper.)

# **2: Data preprocessing**

**🎵For Audio**: The raw audio files were very long and contained extended periods of silence. We used a VAD (Voice Activity Detection) model to identify and remove these silences. The VAD model detected speech segments, which I then concatenated to create a single continuous audio without silence. 

Particularly I used **Silero VAD** model and script for getting speech segment from VAD can be found [here](https://github.com/Amg9794/ML_Assignment_ASR/blob/main/data_preparation/vad_speech_seg_extracter.py)  and script for concatenating VAD output from [here](https://github.com/Amg9794/ML_Assignment_ASR/blob/main/data_preparation/concatenate_vad_outoput.py) .

**Original Signal (5 hr)**

![Org signal (5hr)](https://github.com/Amg9794/ML_Assignment_ASR/blob/main/data_preparation/Org.png)

Speech Segments labels

![Speech Segments labels](https://github.com/Amg9794/ML_Assignment_ASR/blob/main/data_preparation/segmneted%20view.png)

more closer look

![close look](https://github.com/Amg9794/ML_Assignment_ASR/blob/main/data_preparation/close%20look.png)

Final Signal after removing Silence (2 h 45 min)

![after vad](https://github.com/Amg9794/ML_Assignment_ASR/blob/main/data_preparation/after%20vad.png)

**🗒️For Text :** This was bit challenging due to inconsistencies in pdf. Each pdf contain Speaker detail and corresponding dialogues. 

few challenges that i faced :

- Dealing with placeholder [UNCLEAR] and other kind of placeholder.

Here I performed following steps to get metadata from pdf and saved in json file following this below structure :

***S1: Identifying Speaker Lines***

Through manual inspection of several PDFs, I found a consistent pattern: speaker names followed by a colon (:) and their dialogue [SPEAKER: dialogue]. I wrote the is_valid_speaker_line function in this script to check for these specific patterns, which are detailed in the script.

> Note: Speaker names were inconsistent across PDFs, making it difficult to determine the exact number of unique speakers, though they could be identified manually. For example, "CHIEF JUSTICE D. Y. CHANDRACHUD" appears in one PDF while "CJ D_Y_CHANDRACHUD" appears in another.
> 

***S2: Extracting Dialogues from PDFs***

a) For each page, I extracted text and split it into lines.

b) For each valid speaker line, I saved the speaker's dialogue (if any) after cleaning.

Step 3: I performed text processing while maintaining timestamps, date formats, and numbers as they were. Finally, I generated a JSON file containing speakers and their dialogues in this format

```json
[
    {
        "real_speaker": "CHIEF JUSTICE D. Y. CHANDRACHUD",
        "dialogue": "i'll take notice of it"
    },
    {
        "real_speaker": "PETITIONER'S COUNSEL",
        "dialogue": "my lord this is a demolition matter"
    }
   ]
```

**challenges** 

- Since pdf were inconsistent in writing few terms like Dates, Volume numbers (2 or ii) so finding clear pattern without manual inspection was difficult.
- Different pdfs have different placeholder like [unclear] , [inaudible] , [no audio] and many such others so if we don’t have enough information about these we may add or miss something in dialogue that may cause issue during alignments.
- For ex : [UNCLEAR] - it means something is being spoken but not sure about which word so if we remove this placeholder from dialogue then during alignment at this time frame word boundaries of other word may affected as we do not have word for this frame and aligner most likely try to align next word at this time frame which may be not correct timestamp of that word.

# **3- Data Preparation : To get aligned data**

The next task was to align each speaker's dialogue (text) with their corresponding audio segments.

**Approach -1** 

I initially experimented with speaker diarization to extract speech segments, utilizing the [pyannote/speaker-diarization-3.1 model.](https://huggingface.co/pyannote/speaker-diarization-3.1) The resulting segments were obtained as follows:

![overalp speech.png](https://github.com/Amg9794/ML_Assignment_ASR/blob/main/data_preparation/overalp%20speech.png)

I identified the following issues with this approach, which led me to explore an alternative method for obtaining speaker segments

- ***Overlapping Segments*:**  The approach produced overlapping segments. Upon reviewing these segments, I found that the overlaps typically consisted of short words or filler words.
- ***Inconsistent Speaker Identification***: The [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) model assigned different speaker IDs to the same segments across multiple runs. Additionally, it failed to accurately identify the total number of speakers in the audio (e.g., detecting only 5 or 6 speakers when the audio contained 7 distinct speakers).

**Approach-2** 

In this approach, I used [NEMO Aligner](https://github.com/NVIDIA/NeMo-Aligner), a scalable toolkit for efficient model alignment.

NEMO Aligner takes an audio file and its text manifest as input, then generates token and word-level alignments for the audio.

![image.png](https://github.com/Amg9794/ML_Assignment_ASR/blob/main/data_preparation/nfa_forced_alignment_pipeline.png)

so for above example manifest file will look like this 

```jsx
{"audio_filepath": "/hdd2/Aman/scripts/ML_assign/Dataset/audio_wav_wo_sil/case1_2023_12_12.wav", "text": "i'll take notice of it | my lord this is a demolition matter”}
```

here I added separator “ | “ to separate dialogue of two different speaker .I specified in the NFA config that `additional_segment_grouping_separator="|"`. This will make sure Force Aligner  produces timestamps for the groups of text that is in between the `|` characters.

Let’s look at  produce segments , words and token level alignments

**Segment level alignment**

```jsx
case1_2023_12_12 1 0.00 0.80 i'll<space>take<space>notice<space>of<space>it NA lex NA
case1_2023_12_12 1 0.80 1.44 my<space>lord<space>this<space>is<space>a<space>demolition<space>matter NA lex NA
```

**Words level alignments**

```jsx
case1_2023_12_12 1 0.00 0.32 i'll NA lex NA
case1_2023_12_12 1 0.32 0.08 take NA lex NA
case1_2023_12_12 1 0.40 0.24 notice NA lex NA
case1_2023_12_12 1 0.64 0.08 of NA lex NA
case1_2023_12_12 1 0.72 0.08 it NA lex NA
case1_2023_12_12 1 0.80 0.08 my NA lex NA
case1_2023_12_12 1 0.88 0.24 lord NA lex NA
case1_2023_12_12 1 1.20 0.08 this NA lex NA
case1_2023_12_12 1 1.36 0.08 is NA lex NA
case1_2023_12_12 1 1.44 0.16 a NA lex NA
case1_2023_12_12 1 1.60 0.48 demolition NA lex NA
case1_2023_12_12 1 2.16 0.08 matter NA lex NA
```

The next task involved audio segmentation and dataset preparation for machine learning tasks.

Here I processes raw audio files, segmented them into smaller chunks (≤ 30 seconds) using segment-level and word-level CTM files, which provide time-aligned transcripts.

Segments are aligned with individual words to ensure precise boundaries and accurate transcriptions.

then segments are matched to specific speakers using JSON metadata, which contains dialogue and speaker information.

Long segments are split into shorter chunks (≤ 30 seconds) based on word timings to make them suitable for machine learning tasks like speech recognition (Whisper = 30 sec)

finally metadata (audio path, transcription, duration) is stored in CSV files for each split.
csv file will like this

```jsx
/hdd2/Aman/scripts/ML_assign/Final_dataset/test/MEHTA/case6_2023_04_10_22_0_7ff5d2d4.wav,yes my lord someone takes a bribe outside the house my lord and claim immunity,3.52
```

**Shortcomings:**

**NeMo performs really well at providing timestamps for words or segments; however, there are a few issues that can impact the final alignment quality.** One key issue arises when the input documents are not parsed properly. In such cases, dialogues may be incorrectly assigned to the wrong speaker during Step 2, leading to a mismatch between the audio and the corresponding text.

Another challenge occurs during the chunking of longer segments into 30-second chunks. This process can sometimes produce very short chunks (e.g., 0.05 or 0.06 seconds), which effectively behave like silence and may not carry meaningful information.

**To ensure label accuracy for training**, we conducted a sanity check by running inference using the Whisper Large V3 pretrained model on our labeled dataset. We filtered out samples with a Word Error Rate (WER) greater than 100, where none of the words or word sequences in the ground truth matched the model's generated hypothesis. These samples clearly indicated incorrectly prepared labels, especially for longer segments where sufficient context exists to make such judgments.

The filtered dataset, containing only reliably aligned samples, was taken as our final dataset for training , validation and testing.

# **Ⓜ️Model Selection and Evaluation**

I primarily used Whisper as it offers high transcription accuracy and handles diverse accents and background noise, making it reliable for legal recordings.

The model was evaluated using the WER , CER metric. however for uses like medical transcription or legal transcription ,ASR systems should be judged by how well they capture intended meaning, not just error rates so we also evaluated our model on SEMA SCORE that assess whether generated hypothesis accurately convey same meaning for spoken utterance or not for same WER.

.

| **Model** | **WER** | **CER** | **SEMA SCORE** |
| --- | --- | --- | --- |
| Whisper Med PT | 21.86 |  16.37 | 79.05 |
| Whisper Med FT (SC hearing) | **14.95** | 10.21 | 87.52 |
| Whisper Large v3 PT | 19.39 | 13.85 | 80.73 |

For  consider example (Sema score)

Ref : -The community center offers programs focused on promoting the well-being of local families

Hyp1 :- The community center offers programs focused on promoting the wellbeing of local families

Hyp2 : - The community center offers programs focused on promoting the velbeing of local families

Here Hyp1 and Hyp2 have same WER for but Hyp1 is more closer to Ref semantically.

# **🔚Conclusion**

In this project, I developed a complete pipeline to preprocess, align, and prepare SC hearing data for fine-tuning ASR models. Starting from noisy raw audio and inconsistent PDF transcripts, I addressed challenges like silence removal, speaker-dialogue alignment, and text normalization. By leveraging tools like Silero VAD and NVIDIA NeMo Aligner, I was able to generate high-quality, time-aligned audio-text pairs suitable for ASR training. These efforts enabled effective fine-tuning of Whisper models and meaningful evaluation using WER, CER, and SEMA scores.

# **Future improvements**

1. **Handling Overlapping Speech**: This implementation did not fully address overlapping speech in the audio segments. AS mentioned previously mostly overlapping segments are very short related to filler words and word like yes, no. so we can use combine both nemo aligner and diarization output to get non overlapping segments.

2- **Noise reduction /Speech enhancement** **:**- I didn’t implement this as well because i wanted to have fair comparison with method that implement this thing as noise reduction comes at cost of distorting the speech signal itself. In future this can be implemented which van help in getting more accurate speech segments from VAD and precise alignment from aligner.

3- **Chunking :** This can be optimized further more to handle shorter segments of words like yes, no , sorry.

4- **Model Training :-** Next we can try model training by proving previous transcription as prompt . It surely help model to understand the context specially for handling domain specific term like Article 32(a) , volume 6A(b).

5- **Reducing Manual work :**  Since the quality of the aligned data also depends on how well we extract the relevant text from the PDFs, identifying known patterns—such as placeholders or speaker identities—can greatly reduce the need for manual inspection of each page. This can make the overall processing faster and more efficient.
