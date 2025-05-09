'''
This saves the Speaker and corresponds dialogues from the PDF files in a JSON format.

'''

import pdfplumber
import re
import inflect
import os
import json
import unicodedata
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Suppress pdfminer warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)
inflect_engine = inflect.engine()

# Contractions dictionary (Here i included those which i think are useful after checking the audio the way they are being spoken)
contractions_dict = {
 
    "aren't": "are not",
    "I'm": "I am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "there's": "there is",
    "I've": "I have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "he's": "he is",
    "she's": "she is",
    "you're": "you are",
    "i'll": "i will",
    "you'll": "you will",
    "we'll" : "we will",
}

filler_words = ["um", "uh", "you know"]

# Expand contractions
def expand_contractions(text, contractions_dict):
    pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')
    return pattern.sub(lambda x: contractions_dict[x.group()], text)

# Remove filler words
def remove_filler_words(text, filler_words):
    pattern = re.compile(r'\b(' + '|'.join(filler_words) + r')\b', re.IGNORECASE)
    return pattern.sub('', text).strip()

# Normalize Unicode characters and punctuation
def normalize_text(text):
    text = re.sub(r'\\u[0-9a-fA-F]{4}', lambda m: chr(int(m.group(0)[2:], 16)), text)
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    # Fix the character class patterns by using individual Unicode code points
    text = re.sub(r'[\u2018\u2019]', "'", text)  # Fix for single quotes
    text = re.sub(r'[\u201C\u201D]', '"', text)  # Fix for double quotes
    text = re.sub(r'[\u0025\u066a\uff05\u2030\u2031]', '%', text)
    text = re.sub(r'[\u0028\u207d\u208d\uff08]', '(', text)
    text = re.sub(r'[\u0029\u207e\u208e\uff09]', ')', text)
    return text

# Convert percentage symbols
def convert_percentages(text):
    def replace_percent(match):
        num = match.group(1)
        return f"{num} percent "
    text = re.sub(r'\b(\d+)%(?=\s*|$|[,.]?)', replace_percent, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Remove parentheses but keep content to handle citations 
def clean_citation_parentheses(text):
    # Remove parentheses 
    text = re.sub(r'\(', '', text)
    text = re.sub(r'\)', '', text)
    # Handle hyphens contextually: convert number ranges like "1192-1195" to "1192 to 1195"
    text = re.sub(r'(\d+)-(\d+)', r'\1 to \2', text)
    text = re.sub(r'-', ' ', text)  # # Remove remaining hyphens (e.g., in words)
    # Handle citations (e.g., 6A2 → 6A 2, but preserve multi-digit numbers like 1195)
    # Match a number followed by a letter (not a digit), e.g., "6A" → "6 A"
    text = re.sub(r'\b((?:Article|Section|Page)\s+)?(\d+)([a-zA-Z])', r'\1\2 \3', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\d+)([a-zA-Z])', r'\1 \2', text, flags=re.IGNORECASE)
    return text

# Handle [UNCLEAR] and [INAUDIBLE] placeholders (if other present need to filter manully by searching [text])
def process_placeholders(text):
    text = re.sub(r'\[UNCLEAR\]', 'UNCLEAR', text)
    text = re.sub(r'\[INAUDIBLE\]', 'INAUDIBLE', text)
    text = re.sub(r'\[.*?\]', '', text)  # Remove any other placeholders
    return text

# Protect decimal format
def protect_decimal_points(match):
    return match.group().replace(".", "DECIMAL_POINT")

# Protect time formats 
def protect_time_formats(match):
    return match.group().replace(":", "TIME_COLON")

# Enhanced text cleaning to remove punctuation, preserving ordinals, time formats, and decimal points
def clean_punctuation(text):
    # Protect ordinals by ensuring no space is added between number and suffix
    text = re.sub(r'(\d+)\s*(st|nd|rd|th)(?=\s|$)', r'\1\2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{1,2}:\d{2}(:\d{2})?(\s*[AP]M)?', protect_time_formats, text)
    text = re.sub(r'\b\d+\.\d+\b', protect_decimal_points, text)
    
    # Remove punctuation, but avoid affecting protected patterns
    text = re.sub(r'[.,!?;"](?=\s|$|[.,!?;" ])', '', text)  # Remove punctuation at word boundaries
    text = re.sub(r'\.{2,}', '', text)
    text = re.sub(r'[;:]', '', text)  # Remove other colons and semicolons
    text = text.replace("TIME_COLON", ":")
    text = text.replace("DECIMAL_POINT", ".")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Preprocess text
def preprocess_text(text):
    text = normalize_text(text)
    text = expand_contractions(text, contractions_dict)
    text = remove_filler_words(text, filler_words)
    text = convert_percentages(text)
    text = clean_citation_parentheses(text)
    text = process_placeholders(text)
    text = clean_punctuation(text)
    text = text.lower()  # Convert to lowercase
    return text

# Clean dialogue text
def clean_dialogue_text(dialogue):
    dialogue = re.sub(r'\d+$', '', dialogue).strip()
    dialogue = preprocess_text(dialogue)
    return dialogue

# Improved check if a line is a valid speaker line
def is_valid_speaker_line(line):
    line = line.strip()
    if not line or ":" not in line or is_metadata(line) or line.isdigit():
        return False
        
    speaker_part = line.split(":", 1)[0].strip()
    if not speaker_part:
        logger.debug(f"Rejected speaker line (empty): {line}")
        return False
    
    # Accept common speaker patterns like "PETITIONER'S COUNSEL 1:" - must be uppercase
    common_speaker_patterns = [
        r"^PETITIONER['']S COUNSEL \d+$",
        r"^RESPONDENT['']S COUNSEL \d+$",
        r"^JUDGE \d+$",
        r"^JUSTICE \d+$",
        r"^COUNSEL \d+$",
        r"^ATTORNEY \d+$",
        r"^WITNESS \d+$",
        r"^THE COURT$",
        r"^THE WITNESS$"
    ]
    
    for pattern in common_speaker_patterns:
        if re.match(pattern, speaker_part):  # Uppercase required - no IGNORECASE flag
            return True
    
    # If not a common pattern, check if all alphabetic characters are uppercase
    alphabetic_chars = ''.join(char for char in speaker_part if char.isalpha())
    if not alphabetic_chars or alphabetic_chars != alphabetic_chars.upper():
        logger.debug(f"Rejected speaker line (not all alphabetic chars uppercase): {line}")
        return False
        
    # # Additional validation to ensure it's a proper speaker line:
    # # 1. Speaker names are typically short (less than N words)
    # # 2. Speaker names typically don't have punctuation other than spaces
    # word_count = len(speaker_part.split())
    # if word_count > 5:  # Speaker names are usually 1-5 words
    #     logger.debug(f"Rejected speaker line (too many words in speaker): {line}")
    #     return False
        
    # # Only allow letters, numbers, spaces, apostrophes, and limited punctuation in speaker names
    # if re.search(r'[^\w\s.\'-]', speaker_part):
    #     logger.debug(f"Rejected speaker line (invalid characters in speaker): {line}")
    #     return False
        
    return True

# Improved check if a line is a timestamp or metadata
def is_timestamp_or_metadata(line):
    line = line.strip()
    # Only match if the ENTIRE line is a timestamp
    timestamp_patterns = [
        r"^\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?\s*(IST|EST|GMT|UTC)?$",  # Match exact timestamp patterns like "11:00 AM IST"
        r"^\d{1,2}:\d{2}(:\d{2})?$"  # Simple time format with nothing else
    ]
    
    # Return True only if the ENTIRE line matches a timestamp pattern or is metadata
    return any(re.match(pattern, line) is not None for pattern in timestamp_patterns) or is_metadata(line)

# Check if a line is metadata
def is_metadata(line):
    metadata_patterns = [
        r"Transcribed by TERES",
        r"END OF DAY.?S PROCEEDINGS",
        r"^\d+$",
        r"END OF THIS PROCEEDING"  # Added new metadata pattern
    ]
    return any(re.search(pattern, line.strip(), re.IGNORECASE) for pattern in metadata_patterns)

# Clean speaker names
def clean_speaker_name(speaker):
    return re.sub(r'\d+', '', speaker).strip()

# Remove line numbers at the start of a line
def remove_line_number(line):
    return re.sub(r'^\d+\s+', '', line).strip()

# Extract dialogues from PDF
def extract_dialogues_from_pdf(pdf_path):
    dialogues = []
    current_speaker = None
    current_dialogue = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[1:]:  # Skip first page (metadata)
            text = page.extract_text()
            if text:
                lines = text.split("\n")
                for line in lines:
                    line = line.strip()
                    # Skip only if the entire line is a timestamp or metadata
                    if is_timestamp_or_metadata(line):
                        continue
                    line = remove_line_number(line)
                    if not line:
                        continue
                    if is_valid_speaker_line(line):
                        # Save previous speaker's dialogue
                        if current_speaker and current_dialogue:
                            full_dialogue = " ".join(current_dialogue).strip()
                            cleaned_dialogue = clean_dialogue_text(full_dialogue)
                            # Skip if dialogue is only UNCLEAR or INAUDIBLE
                            if cleaned_dialogue not in ['unclear', 'inaudible'] and not is_metadata(full_dialogue):
                                dialogues.append({
                                    "real_speaker": clean_speaker_name(current_speaker).upper(),  # Convert speaker to uppercase
                                    "dialogue": cleaned_dialogue
                                })

                        # Start new speaker
                        speaker, dialogue = line.split(":", 1)
                        current_speaker = clean_speaker_name(speaker.strip())
                        dialogue = dialogue.strip()
                        current_dialogue = [dialogue] if dialogue else []

                    elif current_speaker and line and not is_metadata(line):
                        current_dialogue.append(line)

        # Save the last dialogue
        if current_speaker and current_dialogue:
            full_dialogue = " ".join(current_dialogue).strip()
            cleaned_dialogue = clean_dialogue_text(full_dialogue)
            if cleaned_dialogue not in ['unclear', 'inaudible'] and not is_metadata(full_dialogue):
                dialogues.append({
                    "real_speaker": clean_speaker_name(current_speaker).upper(),  # Convert speaker to uppercase
                    "dialogue": cleaned_dialogue
                })

    return dialogues

# Process a single file
def process_single_file(case_name, pdf_folder, output_json_folder):
    pdf_path = os.path.join(pdf_folder, f"{case_name}.pdf")
    output_json_path = os.path.join(output_json_folder, f"{case_name}.json")

    if not os.path.exists(output_json_folder):
        os.makedirs(output_json_folder)

    if os.path.exists(pdf_path):
        print(f"Processing {case_name}...")
        dialogues = extract_dialogues_from_pdf(pdf_path)
        
        # Create JSON output with only real_speaker and dialogue
        result = [{"real_speaker": d["real_speaker"], "dialogue": d["dialogue"]} for d in dialogues]
        
        with open(output_json_path, "w") as output_file:
            json.dump(result, output_file, indent=4)
        print(f"Output saved to: {output_json_path}")
    else:
        print(f"Error: PDF ({pdf_path}) file not found!")

# Process all files in a directory
def process_all_files_in_directory(pdf_folder, output_json_folder):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_json_folder):
        os.makedirs(output_json_folder)
    
    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_folder}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF file
    for pdf_file in pdf_files:
        case_name = os.path.splitext(pdf_file)[0]  # Remove file extension
        print(f"Processing {case_name}...")
        
        pdf_path = os.path.join(pdf_folder, pdf_file)
        output_json_path = os.path.join(output_json_folder, f"{case_name}.json")
        
        dialogues = extract_dialogues_from_pdf(pdf_path)
        
        # Create JSON output with only real_speaker and dialogue
        result = [{"real_speaker": d["real_speaker"], "dialogue": d["dialogue"]} for d in dialogues]
        
        with open(output_json_path, "w") as output_file:
            json.dump(result, output_file, indent=4)
        print(f"Output saved to: {output_json_path}")
    
    print(f"Completed processing all PDF files!")

# Paths configuration
pdf_folder = '/hdd2/Aman/scripts/ML_assign/Dataset/raw/transcripts_pdfs'
output_json_folder = '/hdd2/Aman/scripts/ML_assign/spk_dialg_json'

# Process all PDF files in the directory
process_all_files_in_directory(pdf_folder, output_json_folder)
print("Processed all PDF files in the directory!")