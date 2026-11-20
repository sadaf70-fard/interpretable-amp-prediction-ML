import os
import numpy as np
import pandas as pd
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis

property_table = {
    "Hydrophobicity index": {'A': -0.21, 'R': 2.11, 'N': 0.96, 'D': 1.36, 'C': -6.04, 'Q': 1.52, 'E': 2.3, 'G': 0, 'H': -1.23, 'I': -4.81, 'L': -4.68, 'K': 3.88, 'M': -3.66, 'F': -4.65, 'P': 0.75, 'S': 1.74, 'T': 0.78, 'W': -3.32, 'Y': -1.01, 'V': -3.5},
    "Hydrophilicity scale": {'A': 0.78, 'R': 1.58, 'N': 1.2, 'D': 1.35, 'C': 0.55, 'Q': 1.19, 'E': 1.45, 'G': 0.68, 'H': 0.99, 'I': 0.47, 'L': 0.56, 'K': 1.1, 'M': 0.66, 'F': 0.47, 'P': 0.69, 'S': 1.0, 'T': 1.05, 'W': 0.7, 'Y': 1.0, 'V': 0.51},
    "Buriability": {'A': 13.4, 'R': 8.5, 'N': 7.6, 'D': 8.2, 'C': 22.6, 'Q': 8.5, 'E': 7.3, 'G': 7.0, 'H': 11.3, 'I': 20.3, 'L': 20.8, 'K': 6.1, 'M': 15.7, 'F': 23.9, 'P': 9.9, 'S': 8.2, 'T': 10.3, 'W': 24.5, 'Y': 19.5, 'V': 19.5},
    "Residue volume": {'A': 52.6, 'R': 109.1, 'N': 75.7, 'D': 68.4, 'C': 68.3, 'Q': 89.7, 'E': 84.7, 'G': 36.3, 'H': 91.9, 'I': 102.0, 'L': 102.0, 'K': 105.1, 'M': 97.7, 'F': 113.9, 'P': 73.6, 'S': 54.9, 'T': 71.2, 'W': 135.4, 'Y': 116.2, 'V': 85.1},
    "Polarity": {'A': 0.0, 'R': 52.0, 'N': 3.38, 'D': 49.7, 'C': 1.48, 'Q': 3.53, 'E': 49.9, 'G': 0.0, 'H': 51.6, 'I': 0.13, 'L': 0.13, 'K': 49.5, 'M': 1.43, 'F': 0.35, 'P': 1.58, 'S': 1.67, 'T': 1.66, 'W': 2.1, 'Y': 1.61, 'V': 0.13},
    "Localized electrical effect": {'A': -0.01, 'R': 0.04, 'N': 0.06, 'D': 0.15, 'C': 0.12, 'Q': 0.05, 'E': 0.07, 'G': 0.0, 'H': 0.08, 'I': -0.01, 'L': -0.01, 'K': 0.0, 'M': 0.04, 'F': 0.03, 'P': 0.0, 'S': 0.11, 'T': 0.04, 'W': 0.0, 'Y': 0.03, 'V': 0.01},
    "Alpha-CH chemical shifts": {'A': 4.35, 'R': 4.38, 'N': 4.75, 'D': 4.76, 'C': 4.65, 'Q': 4.37, 'E': 4.29, 'G': 3.97, 'H': 4.63, 'I': 3.95, 'L': 4.17, 'K': 4.36, 'M': 4.52, 'F': 4.66, 'P': 4.44, 'S': 4.5, 'T': 4.35, 'W': 4.7, 'Y': 4.6, 'V': 3.95},
    "Alpha-helix indices": {'A': 1.29, 'R': 1.0, 'N': 0.81, 'D': 1.1, 'C': 0.79, 'Q': 1.07, 'E': 1.49, 'G': 0.63, 'H': 1.33, 'I': 1.05, 'L': 1.31, 'K': 1.33, 'M': 1.54, 'F': 1.13, 'P': 0.63, 'S': 0.78, 'T': 0.77, 'W': 1.18, 'Y': 0.71, 'V': 0.81},
    "Beta-strand indices": {'A': 0.84, 'R': 1.04, 'N': 0.66, 'D': 0.59, 'C': 1.27, 'Q': 1.02, 'E': 0.57, 'G': 0.94, 'H': 0.81, 'I': 1.29, 'L': 1.1, 'K': 0.86, 'M': 0.88, 'F': 1.15, 'P': 0.8, 'S': 1.05, 'T': 1.2, 'W': 1.15, 'Y': 1.39, 'V': 1.56},
    "Aperiodic indices": {'A': 0.91, 'R': 1.0, 'N': 1.64, 'D': 1.4, 'C': 0.93, 'Q': 0.94, 'E': 0.97, 'G': 1.51, 'H': 0.9, 'I': 0.65, 'L': 0.59, 'K': 0.82, 'M': 0.58, 'F': 0.72, 'P': 1.66, 'S': 1.23, 'T': 1.04, 'W': 0.67, 'Y': 0.92, 'V': 0.6},
    "Relative mutability": {'A': 100, 'R': 65, 'N': 134, 'D': 106, 'C': 20, 'Q': 93, 'E': 102, 'G': 49, 'H': 66, 'I': 96, 'L': 40, 'K': 56, 'M': 94, 'F': 41, 'P': 56, 'S': 120, 'T': 97, 'W': 18, 'Y': 41, 'V': 74},
    "Optical rotation": {'A': 1.8, 'R': 12.5, 'N': -5.6, 'D': 5.05, 'C': -16.5, 'Q': 6.3, 'E': 12, 'G': 0, 'H': -38.5, 'I': 12.4, 'L': -11, 'K': 14.6, 'M': -10, 'F': -34.5, 'P': -86.2, 'S': -7.5, 'T': -28, 'W': -33.7, 'Y': -10, 'V': 5.63},
    "Graph shape index": {'A': 1.28, 'R': 2.34, 'N': 1.6, 'D': 1.6, 'C': 1.77, 'Q': 1.56, 'E': 1.56, 'G': 0, 'H': 2.99, 'I': 4.19, 'L': 2.59, 'K': 1.89, 'M': 2.35, 'F': 2.94, 'P': 2.67, 'S': 1.31, 'T': 3.03, 'W': 3.21, 'Y': 2.94, 'V': 3.67},
    "Solvation free energy": {'A': 0.67, 'R': -2.1, 'N': -0.6, 'D': -1.2, 'C': 0.38, 'Q': -0.22, 'E': -0.76, 'G': 0, 'H': 0.64, 'I': 1.9, 'L': 1.9, 'K': -0.57, 'M': 2.4, 'F': 2.3, 'P': 1.2, 'S': 0.01, 'T': 0.52, 'W': 2.6, 'Y': 1.6, 'V': 1.5},
    "Net charge": {'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'Q': 0, 'E': -1, 'G': 0, 'H': 0, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0},
    "HPLC parameter": {'A': 2.1, 'R': 4.2, 'N': 7.0, 'D': 10.0, 'C': 1.4, 'Q': 6.0, 'E': 7.8, 'G': 5.7, 'H': 2.1, 'I': -8.0, 'L': -9.2, 'K': 5.7, 'M': -4.2, 'F': -9.2, 'P': 2.1, 'S': 6.5, 'T': 5.2, 'W': -10.0, 'Y': -1.9, 'V': -3.7}
}

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

# Read sequences from a txt file
def read_sequences(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_id = None
    current_seq = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if current_id and current_seq:
                sequences.append((current_id, ''.join(current_seq)))
            current_id = line[1:].strip()
            current_seq = []
        else:
            current_seq.append(line.strip())

    if current_id and current_seq:
        sequences.append((current_id, ''.join(current_seq)))

    return sequences


# Amino Acid Composition (AAC)
def compute_aac(sequence):
    L = len(sequence)
    counts = Counter(sequence)
    return [counts[aa] / L if L > 0 else 0 for aa in AMINO_ACIDS]

# Dipeptide Composition (DPC)
def compute_dpc(sequence):
    if len(sequence) < 2:
        return [0] * 400
    dipeptides = [a + b for a in AMINO_ACIDS for b in AMINO_ACIDS]
    total = len(sequence) - 1
    counts = Counter([sequence[i:i + 2] for i in range(total)])
    return [counts[dp] / total if total > 0 else 0 for dp in dipeptides]

#  Tripeptide Composition (TPC)
def compute_tpc(sequence):
    if len(sequence) < 3:
        return [0] * 8000
    tripeptides = [a + b + c for a in AMINO_ACIDS for b in AMINO_ACIDS for c in AMINO_ACIDS]
    total = len(sequence) - 2
    counts = Counter([sequence[i:i + 3] for i in range(total)])
    return [counts[tp] / total if total > 0 else 0 for tp in tripeptides]

#  Binary Encoding (One-hot)
def one_hot_encode(seq, max_len=30):
    aa_to_index = {a: i for i, a in enumerate(AMINO_ACIDS)}
    mat = np.zeros((max_len, 20))
    for i, c in enumerate(seq[:max_len]):
        if c in aa_to_index:
            mat[i, aa_to_index[c]] = 1
    return mat.flatten()

#  Physicochemical Properties
def compute_pcp(sequence, table=property_table):
    features = []

    for prop_name, aa_values in table.items():
        values = []
        for aa in sequence:
            if aa in aa_values:
                values.append(aa_values[aa])
        avg = sum(values) / len(values) if values else 0
        features.append(avg)

    return features

# Extract features from all .txt files in a folder
def extract_features_from_folder(folder_path, label=1, max_len=30):
    all_features = []
    all_labels = []
    all_sequences = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            print(f" Processing file: {file_path}")
            sequences = read_sequences(file_path)

            for peptide_id, sequence in sequences:
                try:
                    sequence = sequence.upper()
                    if not sequence.isalpha() or any(aa not in AMINO_ACIDS for aa in sequence):
                        print(f"Invalid or non-standard sequence: {peptide_id}")
                        continue

                    aac = compute_aac(sequence)
                    dpc = compute_dpc(sequence)
                    tpc = compute_tpc(sequence)
                    pcp = compute_pcp(sequence)
                    be = one_hot_encode(sequence, max_len)

                    features = aac + dpc + tpc + pcp + list(be)
                    all_features.append(features)
                    all_labels.append(label)
                    all_sequences.append(sequence)

                except Exception as e:
                    print(f" Error processing sequence {peptide_id}: {e}")

    return np.array(all_features), np.array(all_labels), all_sequences


def main():
    amp_folder = os.path.join("data", "AMP")
    non_amp_folder = os.path.join("data", "Non_AMP")

    print("Extracting AMP features...")
    amp_features, amp_labels, amp_seqs = extract_features_from_folder(amp_folder, label=1)

    print("Extracting non-AMP features...")
    non_amp_features, non_amp_labels, non_amp_seqs = extract_features_from_folder(non_amp_folder, label=0)

    all_features = np.vstack([amp_features, non_amp_features])
    all_labels = np.concatenate([amp_labels, non_amp_labels])
    all_sequences = amp_seqs + non_amp_seqs

    df = pd.DataFrame(all_features)
    df['sequence'] = all_sequences
    df['label'] = all_labels
    df.to_csv('amp_nonamp_combined.csv', index=False)

    print("All features extracted and saved to 'amp_nonamp_combined.csv'")


if __name__ == '__main__':
    main()