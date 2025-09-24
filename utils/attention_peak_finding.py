import pandas as pd
import torch
import numpy as np
from scipy.stats import entropy
import sklearn
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from scipy.signal import find_peaks
from scipy.fft import fft, ifft
from scipy.signal import find_peaks
from tqdm.auto import tqdm
import os
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint as IP
import argparse

def rescale_att_protein(prot_idx, att_vec, hou):
    # get protein length
    len_prot = len(hou.sequence.iloc[prot_idx])
    if len_prot < 1024:
        truncated_att = att_vec[:len_prot]
    else:
        truncated_att = att_vec[:1024]
    mm_att = sklearn.preprocessing.MinMaxScaler().fit_transform(truncated_att.reshape(-1, 1)).flatten()
    z_att = sklearn.preprocessing.StandardScaler().fit_transform(truncated_att.reshape(-1, 1)).flatten()
    # add back the padding
    if len(mm_att) < 1024:
        diff_len = 1024 - len_prot
        pad_0 = np.zeros(diff_len)
        mm_att = np.concatenate([mm_att, pad_0])
        z_att = np.concatenate([z_att, pad_0])
    return mm_att, z_att
    

# Peak calling
def peak_caller(att_vec, thresh=0.5, window_size=5, height=1.0):
    """
    Peak caller function that smooths attention vector and finds peaks with start/end indices.
    
    Args:
        att_vec: attention vector (1D numpy array)
        thresh: threshold for peak detection (default 0.5)
        window_size: window size for moving average smoothing (default 10)
    
    Returns:
        dict with:
            - 'peaks': peak indices in original coordinate system
            - 'peak_starts': start indices of peaks in original coordinate system  
            - 'peak_ends': end indices of peaks in original coordinate system
            - 'smoothed_signal': the smoothed attention vector (shorter than original)
    """
    original_length = len(att_vec)
    
    # Apply moving average smoothing
    # Using np.convolve with 'valid' mode - output length will be (original_length - window_size + 1)
    smoothed_att = np.convolve(att_vec, np.ones(window_size)/window_size, mode='valid')
    smoothed_att = np.where(smoothed_att < thresh, 0, smoothed_att)
    # Find peaks in smoothed signal
    peaks, peak_properties = find_peaks(smoothed_att,height=height)
    
    # For each peak, find start and end indices by looking for where signal drops below threshold
    peak_starts = []
    peak_ends = []
    
    for peak_idx in peaks:
        # Find start: go left from peak until signal drops below threshold or reach beginning
        start_idx = peak_idx
        while start_idx > 0 and smoothed_att[start_idx - 1] >= thresh:
            start_idx -= 1
            
        # Find end: go right from peak until signal drops below threshold or reach end
        end_idx = peak_idx
        while end_idx < len(smoothed_att) - 1 and smoothed_att[end_idx + 1] >= thresh:
            end_idx += 1
            
        peak_starts.append(start_idx)
        peak_ends.append(end_idx)
    
    # Transform coordinates back to original attention vector coordinate system
    # The smoothed signal starts at position (window_size-1)//2 in the original signal
    offset = (window_size - 1) // 2
    
    # Map smoothed coordinates to original coordinates
    original_peaks = [p + offset for p in peaks]
    original_starts = [s + offset for s in peak_starts]  
    original_ends = [e + offset for e in peak_ends]
    
    # Ensure coordinates don't exceed original vector bounds
    original_peaks = [min(p, original_length - 1) for p in original_peaks]
    original_starts = [max(0, min(s, original_length - 1)) for s in original_starts]
    original_ends = [min(e, original_length - 1) for e in original_ends]
    
    return {
        'peaks': np.array(original_peaks),
        'peak_starts': np.array(original_starts),
        'peak_ends': np.array(original_ends),
        'smoothed_signal': smoothed_att
    }
    
def grab_actual_seq(protein_idx, hou):
    seq = hou.iloc[protein_idx]['sequence']
    if len(seq) > 1024:
        seq = seq[:512] + seq[-512:]
    return seq

def extract_peak_seqs(protein_idx, hou, z_attentions, length_thresh=5, inflate_to=50):
    seq = grab_actual_seq(protein_idx, hou)
    z_att = z_attentions[protein_idx]
    res = peak_caller(z_att, thresh=0.5, window_size=5)
    peak_binary = np.zeros(len(seq))
    for i in range(len(res['peaks'])):
        peak_binary[res['peak_starts'][i]:res['peak_ends'][i]] = 1
    
    seq_regions = []
    tmp_str = ''
    peak_started = False
    new_peak_indices = []
    tmp_index = [0, 0]
    
    for i, s in enumerate(seq):
        is_in_peak = True if peak_binary[i] == 1 else False
        if is_in_peak == 1 and peak_started == False:
            # start of peak
            peak_started = True
            tmp_index[0] = i
            tmp_str += s
        elif is_in_peak == 1 and peak_started == True:
            # in peak
            tmp_str += s
        elif is_in_peak == 0 and peak_started == True:
            # end of peak
            peak_started = False
            seq_regions.append(tmp_str)
            tmp_str = ''
            tmp_index[1] = i - 1  # Fix: end should be i-1 since i is the first non-peak position
            new_peak_indices.append([tmp_index[0], tmp_index[1]])  # Fix: create new list
            tmp_index = [0, 0]
        else:
            continue
    
    # Handle case where sequence ends while in a peak
    if peak_started:
        seq_regions.append(tmp_str)
        tmp_index[1] = len(seq) - 1
        new_peak_indices.append([tmp_index[0], tmp_index[1]])
    
    filtered_seq_regions = []
    for n, s in enumerate(seq_regions):
        if len(s) < length_thresh:
            continue
        if len(s) < inflate_to:
            # Fix: use correct indexing and add bounds checking
            peak_start = new_peak_indices[n][0]  # Fix: use [n][0] instead of [0]
            peak_length = len(s)
            extension_needed = inflate_to - peak_length
            
            # Calculate new boundaries with bounds checking
            new_start = max(0, peak_start - int(extension_needed / 2))
            new_end = min(len(seq), new_start + inflate_to)
            
            # Adjust start if we hit the end boundary
            if new_end - new_start < inflate_to:
                new_start = max(0, new_end - inflate_to)
            
            extended_seq = seq[new_start:new_end]
            filtered_seq_regions.append(extended_seq)
        else:
            filtered_seq_regions.append(s)
    
    return filtered_seq_regions  # Fix: return filtered_seq_regions instead of seq_regions
    
def get_attention_peak_positions(hou, z_attentions):
    print("Getting attention peak positions")
    for idx, row in tqdm(hou.iterrows(), total=len(hou)):
        peak_seqs = extract_peak_seqs(idx, hou, z_attentions)
        positions = []
        for peak_seq in peak_seqs:
            peak_start = row['sequence'].find(peak_seq)
            peak_end = peak_start + len(peak_seq)
            positions.append([peak_start, peak_end])
        hou.at[idx, 'peak_positions'] = str(positions)
    return hou

def main(args):
    hou = pd.read_csv(args.input_file)
    experiment_folder = args.best_model
    output_file = args.output_file

    best_model_metrics = pd.read_csv(f'{experiment_folder}/overall_test_metrics.csv')
    all_thresholds = np.load(f'{experiment_folder}/all_thresholds.npy')

    test_fold_results = []
    test_attentions = []
    unique_proteins = []
    for i in range(0,5):
        folder = f'{experiment_folder}/fold_{i}/'
        f = pd.read_csv(f'{folder}fold_{i}_test_predictions.csv')
        f['fold'] = i
        test_fold_results.append(f)
        att = torch.load(f'{folder}fold_{i}_test_attention.pt')
        test_attentions.append(att)
        print(f.shape)
        print(att.shape)

    test_fold_results = pd.concat(test_fold_results)
    test_attentions = torch.stack(test_attentions).mean(dim=0).cpu().numpy()

    lengths = []
    for idx, row in hou.iterrows():
        seq = row['sequence']
        lengths.append(len(seq))
    hou.loc[:, 'Length'] = lengths
    
    
    mm_attentions = []
    z_attentions = []
    for i in tqdm(range(len(hou))):
        mm_att, z_att = rescale_att_protein(i, test_attentions[i], hou)
        mm_attentions.append(mm_att)
        z_attentions.append(z_att)
    mm_attentions = np.array(mm_attentions)
    z_attentions = np.array(z_attentions)
    
    all_loc_classes = []
    for idx, row in hou.iterrows():
        classes = row['level1'].split(';')
        classes = [c.strip() for c in classes if c.strip()]
        all_loc_classes.extend(classes)
    all_loc_classes = list(np.unique(all_loc_classes))
    encoded_loc_vectors = np.zeros((len(hou), len(all_loc_classes)))
    for idx, row in hou.iterrows():
        classes = row['level1'].split(';')
        classes = [c.strip() for c in classes if c.strip()]
        for c in classes:
            if c in all_loc_classes:
                encoded_loc_vectors[idx, all_loc_classes.index(c)] = 1
                
    hou = get_attention_peak_positions(hou, z_attentions)
    hou.to_csv(output_file, index=False)
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Add uniprot canonical sequences (and uniprot ids if needed) to a metadata CSV based on ensmble-gene/uniprot-protein IDs.")
    argparser.add_argument("--input_file", required=True, help="Path to csv with gene/protein IDs")
    argparser.add_argument("--best_model", required=True, help="path to directory for te best model")
    argparser.add_argument("--output_file", required=True, help="path to directory for te best model")
    args = argparser.parse_args()
    main(args)
    