# compute cer from csv
import os
import numpy as np
import sys
import pandas as pd
import argparse

def build_diff(ref, hyp, path):
    result = []
    ref = list(map(lambda x: x.lower(), ref))
    hyp = list(map(lambda x: x.lower(), hyp))
    r_record = -1
    h_record = -1

    for rpointer, hpointer in path:
        if rpointer!=r_record+1 or hpointer!=h_record+1:
            r_buffer = ' '.join(ref[r_record+1:rpointer])
            r_buffer = r_buffer if len(r_buffer)>0 else "*"
            h_buffer = ' '.join(hyp[h_record+1:hpointer])
            h_buffer = h_buffer if len(h_buffer)>0 else "*"
            result.append(f"({r_buffer}->{h_buffer})")

        result.append(ref[rpointer])
        r_record = rpointer
        h_record = hpointer

    if r_record<len(ref)-1 or h_record<len(hyp)-1:
        r_buffer = ' '.join(ref[r_record+1:])
        r_buffer = r_buffer if len(r_buffer)>0 else "*"
        h_buffer = ' '.join(hyp[h_record+1:])
        h_buffer = h_buffer if len(h_buffer)>0 else "*"
        result.append(f"({r_buffer}->{h_buffer})")
    return ' '.join(result)

def compute_cer_from_csv(csv_file, cer_detail_file):
    rst = {
        'Char': 0,
        'Corr': 0,
        'Ins': 0,
        'Del': 0,
        'Sub': 0,
        'Snt': 0,
        'Err': 0.0,
        'wrong_chars': 0,
        'wrong_sentences': 0
    }

    # Read CSV file
    df = pd.read_csv(csv_file)

    # Check for non-string values in the hypothesis and reference columns
    for idx, (hyp_text, ref_text) in enumerate(zip(df.iloc[:, 2], df.iloc[:, 3])):
        if not isinstance(hyp_text, str):
            print(f"Issue in hypothesis at index {idx}: value = {hyp_text} (type = {type(hyp_text)})")
        if not isinstance(ref_text, str):
            print(f"Issue in reference at index {idx}: value = {ref_text} (type = {type(ref_text)})")

    # Create dictionaries from DataFrame
    hyp_dict = dict(zip(df.iloc[:, 1], [list(text) if isinstance(text, str) else [] for text in df.iloc[:, 2]]))
    ref_dict = dict(zip(df.iloc[:, 1], [list(text) if isinstance(text, str) else [] for text in df.iloc[:, 3]]))

    cer_detail_writer = open(cer_detail_file, 'w')
    for hyp_key in hyp_dict:
        if hyp_key in ref_dict:
            out_item = compute_cer_by_line(hyp_dict[hyp_key], ref_dict[hyp_key])
            
            rst['Char'] += out_item['nchars']
            rst['Corr'] += out_item['cor']
            rst['wrong_chars'] += out_item['wrong']
            rst['Ins'] += out_item['ins']
            rst['Del'] += out_item['del']
            rst['Sub'] += out_item['sub']
            rst['Snt'] += 1
            if out_item['wrong'] > 0:
                rst['wrong_sentences'] += 1
            cer_detail_writer.write(str(hyp_key) + print_cer_detail(out_item) + '\n')
            cer_detail_writer.write("ref:" + '\t' + "".join(ref_dict[hyp_key]) + '\n')
            cer_detail_writer.write("hyp:" + '\t' + "".join(hyp_dict[hyp_key]) + '\n')
            cer_detail_writer.write("diff:" + '\t' + build_diff(ref_dict[hyp_key], hyp_dict[hyp_key], out_item['path']) + '\n')

    if rst['Char'] > 0:
        rst['Err'] = round(rst['wrong_chars'] * 100 / rst['Char'], 2)

    print("%CER " + str(rst['Err']))
    cer_detail_writer.write('\n')
    cer_detail_writer.write("%CER " + str(rst['Err']) + " [ " + str(rst['wrong_chars']) + " / " + str(rst['Char']) +
                            ", " + str(rst['Ins']) + " ins, " + str(rst['Del']) + " del, " + str(rst['Sub']) + " sub ]" + '\n')
    cer_detail_writer.write("Scored " + str(len(hyp_dict)) + " sentences, " + str(len(hyp_dict) - rst['Snt']) + " not present in hyp." + '\n')



def compute_cer_by_line(hyp, ref):
    hyp = list(map(lambda x: x.lower(), hyp))
    ref = list(map(lambda x: x.lower(), ref))

    
    len_hyp = len(hyp)
    len_ref = len(ref)

    cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)
    ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)

    for i in range(len_hyp + 1):
        cost_matrix[i][0] = i
    for j in range(len_ref + 1):
        cost_matrix[0][j] = j

    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hyp[i - 1] == ref[j - 1]:
                cost_matrix[i][j] = cost_matrix[i - 1][j - 1]
            else:
                substitution = cost_matrix[i - 1][j - 1] + 1
                insertion = cost_matrix[i - 1][j] + 1
                deletion = cost_matrix[i][j - 1] + 1

                compare_val = [substitution, insertion, deletion]
                min_val = min(compare_val)
                operation_idx = compare_val.index(min_val) + 1
                cost_matrix[i][j] = min_val
                ops_matrix[i][j] = operation_idx

    match_idx = []
    i = len_hyp
    j = len_ref
    rst = {
        'nchars': len_ref,
        'cor': 0,
        'wrong': 0,
        'ins': 0,
        'del': 0,
        'sub': 0,
        'path': []
    }
    
    while i >= 0 or j >= 0:
        i_idx = max(0, i)
        j_idx = max(0, j)

        if ops_matrix[i_idx][j_idx] == 0:  # correct
            if i - 1 >= 0 and j - 1 >= 0:
                match_idx.append((j - 1, i - 1))
                rst['cor'] += 1
            i -= 1
            j -= 1
        elif ops_matrix[i_idx][j_idx] == 2:  # insert
            i -= 1
            rst['ins'] += 1
        elif ops_matrix[i_idx][j_idx] == 3:  # delete
            j -= 1
            rst['del'] += 1
        elif ops_matrix[i_idx][j_idx] == 1:  # substitute
            i -= 1
            j -= 1
            rst['sub'] += 1

        if i < 0 and j >= 0:
            rst['del'] += 1
        elif j < 0 and i >= 0:
            rst['ins'] += 1

    match_idx.reverse()
    wrong_cnt = cost_matrix[len_hyp][len_ref]
    rst['wrong'] = wrong_cnt
    rst['path'] = match_idx

    return rst


def print_cer_detail(rst):
    return ("(" + "nchars=" + str(rst['nchars']) + ",cor=" + str(rst['cor'])
            + ",ins=" + str(rst['ins']) + ",del=" + str(rst['del']) + ",sub="
            + str(rst['sub']) + ") corr:" + '{:.2%}'.format(rst['cor']/rst['nchars'])
            + ",cer:" + '{:.2%}'.format(rst['wrong']/rst['nchars']))


if __name__ == '__main__':
    csv_file = "/hdd2/Aman/scripts/ML_assign/results/english_new_test_PT_WL.csv"
    output_dir = "/hdd2/Aman/scripts/ML_assign/results"
    wer_detail_file = os.path.join(output_dir, 'WL_PT_CER_val')
    compute_cer_from_csv(csv_file, wer_detail_file)    
