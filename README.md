# Interpretable Machine Learning for Antimicrobial Peptide Prediction with Homology-Aware Evaluation and External Validation

Code and data accompanying the manuscript of the same title.

Six classifiers (logistic regression, SVM, random forest, gradient-boosted trees, XGBoost, MLP) are trained on engineered sequence descriptors to separate antimicrobial peptides (AMPs) from non-AMPs. The repository covers the full pipeline: feature extraction, de-duplication, alignment-based clustering, model training under both random and identity-aware splits, SHAP interpretation, and evaluation on an independent external benchmark.

## Contribution note

The initial version of this pipeline — the feature-extraction module and the first implementations of the classifiers — was developed by **Suyash Sanjay Gaikwad** as part of his MSc dissertation at the University of Surrey. The present repository extends that work with sequence de-duplication, alignment-based homology clustering and identity-aware splitting, repeated-seed and bootstrap uncertainty estimates, feature and resampling ablations, threshold and calibration analysis, and external validation on AMPBenchmark.

## Repository contents

| File | Purpose |
|---|---|
| `Feature_Extraction.py` | Feature extraction module: AAC, DPC, TPC, physicochemical properties, binary encoding. Imported by every notebook. |
| `Base.ipynb` | Data preparation: feature extraction, de-duplication, class balancing, alignment-based clustering, column naming, construction of the external benchmark. |
| `Logistic Regression Final.ipynb` | Logistic regression (L2). |
| `Support Vector Machine Final.ipynb` | SVM with RBF kernel. |
| `Random Forest Final.ipynb` | Random forest. |
| `GBT Final.ipynb` | Gradient-boosted trees. |
| `XGBoost Final.ipynb` | XGBoost, plus the bootstrap confidence intervals, feature ablation, threshold and calibration analyses reported in the manuscript. |
| `MLP Final.ipynb` | Multi-layer perceptron. |
| `data/AMP/` | Positive sequences, by source category (APD3). |
| `data/Non_AMP/` | Negative sequences (AMPScanner v2 negative set). |

## Features

Each sequence is represented by 9,036 descriptors:

| Block | Count | Description |
|---|---|---|
| AAC | 20 | Amino acid composition |
| DPC | 400 | Dipeptide composition |
| TPC | 8,000 | Tripeptide composition |
| PCP | 16 | Physicochemical properties, averaged over residues |
| BE | 600 | Binary (one-hot) encoding of the first 30 positions |

Feature selection inside each pipeline reduces this to a small working set; the selected features are listed in `XGBoost Final.ipynb`.

## Running the pipeline

Run `Base.ipynb` first, top to bottom. It produces the intermediate files the model notebooks read:

1. `amp_nonamp_combined.csv` — all sequences with their features (~1.5 GB)
2. `balanced_dataset.csv` — de-duplicated and class-balanced, with cluster assignments
3. `balanced_named.csv` — the same, with descriptive column names
4. `external_benchmark.csv` — the external test set, downloaded from AMPBenchmark and filtered

These files are generated, not committed, because of their size. `Base.ipynb` also prints the homology diagnostic reported in the manuscript: the distribution of each test sequence's maximum identity to the training set.

Each model notebook can then be run independently. Every notebook contains both splitting protocols:

```python
# random split: cluster membership is deliberately ignored
train_idx, test_idx = train_test_split(
    np.arange(len(df)), test_size=0.2, stratify=y, random_state=42)

# each cluster goes entirely to train or entirely to test
#splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#train_idx, test_idx = next(splitter.split(X, y, groups))
```

The random split is active by default and corresponds to the main results. To reproduce the identity-aware results, comment out the first block and uncomment the second.

Model notebooks write figures to `Metrics/` and fitted pipelines to `Saved Models/`; both directories are created automatically.

## Clustering and splitting

Sequences are grouped by pairwise local-alignment identity, computed as the number of identical aligned residues divided by the length of the shorter sequence. A greedy single-representative scheme assigns each sequence, longest first, to the first existing cluster it matches at ≥40% identity, or starts a new cluster. The 6,616 balanced sequences fall into 3,348 clusters (median size 1, maximum 64). `GroupShuffleSplit` then keeps every cluster entirely within either the training or the test set.

## Data sources

- **Positives:** Antimicrobial Peptide Database (APD3), by source category. Note that `Natural.txt` overlaps the other category files, which is why de-duplication is applied before balancing.
- **Negatives:** the negative set distributed with AMPScanner v2.
- **External benchmark:** AMPBenchmark (Sidorczuk et al., 2022), downloaded by `Base.ipynb` from the project's public repository. Positives come from DBAASP; negatives are the Gabere & Noble sampling scheme. Sequences already present in our training data are removed before evaluation.

## Requirements

Python 3.10 or later, with the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

`Base.ipynb` needs internet access to download the external benchmark.

## Citation

If you use this code, please cite the accompanying manuscript. Details will be added here once it is published.
