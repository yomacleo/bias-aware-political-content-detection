# A Bias-Aware Framework for Political Content Detection in Social Media Using Semantic Lexicon Expansion

This repository contains the experimental notebook used to implement and evaluate the methodology presented in the study **“A Bias-Aware Framework for Political Content Detection in Social Media Using Semantic Lexicon Expansion.”**

The project studies how an interpretable semantic lexicon can be expanded to improve the coverage of political-content detection in social media while explicitly measuring the errors introduced by that expansion. The main notebook reproduces the complete experimental workflow, from data preparation and semantic modeling to repeated validation, human evaluation, statistical analysis, and representation benchmarks.

> **Important:** in this project, *bias-aware* refers specifically to **coverage-related under-detection at the political-content selection stage**. It does not imply demographic, ranking, exposure, or downstream recommender-system fairness.

---

## Overview

Political language on social media changes quickly. Fixed keyword lists can miss local expressions, hashtags, names, slogans, and event-specific terminology that were not known when the original lexicon was created.

The notebook explores a transparent alternative based on:

- **Word2Vec** as an inspectable term-level semantic space.
- **Objective k-means clustering** to organize candidate vocabulary.
- Identification of a political semantic reference from predefined seed terms.
- **Centroid-based Top-\(N\) lexicon expansion** using cosine similarity.
- Two controlled semantic scenarios:
  - **Scenario 1:** initial political semantic reference.
  - **Scenario 2:** expanded political semantic reference.
- Weakly supervised classification using the semantic pseudo-labels.
- Leakage-free holdout and repeated cross-validation.
- Independent evaluation against human judgments.
- A separately audited hashtag-based silver standard.
- Representation benchmarks with **TF-IDF, FastText, BETO, and XLM-RoBERTa**.

The goal is not simply to maximize the number of posts labeled as political. The notebook is designed to examine the **trade-off between semantic coverage and classification errors**, especially false positives introduced when the semantic boundary is expanded.

---

## What the Notebook Does

The main notebook, `codigoBIAS_rs_CYG.ipynb`, organizes the complete experiment into a reproducible sequence of stages.

### 1. Environment and reproducibility audit

The notebook checks the Python environment, library versions, available hardware, random seeds, and execution settings. It also generates a runtime requirements file so that the software environment used for the experiments can be documented.

### 2. Dataset loading, cleaning, and auditing

The workflow loads the consolidated X/Twitter datasets and supporting trend information, preserves traceability through stable identifiers, applies the preprocessing protocol, removes empty records, and deduplicates tweets while retaining the political-event context.

The study uses posts associated with two political events in Ecuador:

- **National Strike (June 2022)**
- **Muerte Cruzada (2023)**

The consolidated corpus contains **144,704 tweets before cleaning** and **138,400 tweets after cleaning and deduplication**.

### 3. Independent evaluation resources

The notebook prepares and manages two external evaluation resources:

- A **600-tweet independent human reference**, sampled before semantic model development and labeled by two primary annotators, with disagreements adjudicated by a third annotator.
- A **hashtag-based silver standard** that is independently audited using a human-reviewed stratified sample.

These resources are kept separate from the semantic model-development process.

### 4. Semantic representation and clustering

Word2Vec models are trained using 50-, 100-, and 200-dimensional representations. Candidate terms are normalized and evaluated using objective clustering criteria.

Candidate values of \(k\) are compared with:

- Silhouette score
- Davies–Bouldin index
- Calinski–Harabasz score
- Inertia

HDBSCAN is also evaluated as a density-based comparison, while t-SNE is used only for visualization.

### 5. Semantic lexicon expansion

The political cluster is identified using predefined political seed terms. Its centroid becomes the semantic reference for ranking candidate terms by cosine similarity.

The notebook evaluates controlled Top-\(N\) expansion using:

- Top-25
- Top-50
- Top-100
- Top-200

The primary configuration used in the study is **Top-100**.

This stage makes it possible to inspect which terms are added, how similar they are to the political centroid, and how progressively broader expansion changes political-content coverage.

### 6. Leakage-free classification and validation

All train-dependent components are reconstructed using training data only. This includes:

- Word2Vec training
- candidate-vocabulary construction
- cluster selection
- political-cluster identification
- centroid estimation
- lexicon expansion
- pseudo-label generation
- classifier training

The notebook evaluates Decision Tree, KNN, Gaussian Naive Bayes, and LinearSVC models.

Internal classifier metrics are interpreted as **pseudo-label fidelity**, not as direct political-content accuracy.

### 7. Repeated cross-validation and statistical analysis

Robustness is evaluated with repeated stratified five-fold cross-validation with two repetitions.

The notebook reports fold-level distributions, confidence intervals, and paired comparisons between the two semantic scenarios. Statistical comparisons use paired Wilcoxon signed-rank tests with Holm correction for multiple comparisons.

### 8. External validation

The automatic semantic decisions are evaluated against references that were not used to construct the semantic model.

The notebook includes:

- independent human-ground-truth evaluation;
- hashtag-blind evaluation against the silver standard;
- human auditing of the silver-label procedure;
- confusion-matrix and error-oriented analysis.

This separation is important because strong agreement with automatically generated pseudo-labels does not necessarily imply agreement with human political-content judgments.

### 9. Contemporary representation benchmarks

The notebook compares the Word2Vec-based representation with:

- TF-IDF
- FastText
- BETO
- XLM-RoBERTa

All representations are evaluated under the same weak-supervision setting with a common linear downstream classifier where applicable.

The benchmark considers both predictive behavior and computational cost. In the experiments reported in the paper, BETO and XLM-RoBERTa are evaluated as **frozen representation extractors**, not as fully fine-tuned transformer models.

### 10. Final analysis and export

The final stages consolidate the experimental outputs, descriptive analyses, validation tables, statistical tests, clustering comparisons, Top-\(N\) sensitivity results, benchmark results, and supplementary tables used in the manuscript.

---

## Experimental Workflow

```text
Raw X/Twitter data
        │
        ▼
Cleaning, normalization, deduplication, and audit
        │
        ├──────────────► Independent human reference
        │
        └──────────────► Audited hashtag-based silver standard
        │
        ▼
Leakage-free train / evaluation partitioning
        │
        ▼
Word2Vec semantic representation
        │
        ▼
Objective clustering and political-cluster identification
        │
        ▼
Political centroid
        │
        ├──────────────► Scenario 1: initial semantic reference
        │
        └──────────────► Scenario 2: Top-N semantic expansion
                              │
                              ▼
                     Semantic pseudo-labels
                              │
                              ▼
                      Supervised classifiers
                              │
          ┌───────────────────┼────────────────────┐
          ▼                   ▼                    ▼
 Repeated CV          Human validation      Silver validation
          │                   │                    │
          └───────────────────┼────────────────────┘
                              ▼
                  Statistical and error analysis
                              │
                              ▼
               Representation benchmark and outputs
```

---

## Repository Structure

The notebook expects a project structure similar to:

```text
.
├── codigoBIAS_rs_CYG.ipynb
│
└── outputs_bias_aware/
│
└── requirements_runtime.txt
```

The raw and restricted research data are **not intended to be distributed publicly in this repository**. See [Data Availability](yoma.cleo@hotmail.com).

---

## Environment

The experiments were developed with **Python 3.12.14**.

Main dependencies include:

- NumPy
- pandas
- SciPy
- scikit-learn
- Gensim
- spaCy
- PyTorch
- Transformers
- Bokeh
- OpenPyXL
- psutil

The Spanish spaCy model used by the preprocessing pipeline is:

```bash
python -m spacy download es_core_news_sm
```

A typical environment can be prepared with:

```bash
python -m pip install numpy pandas scipy scikit-learn gensim bokeh openpyxl \
    transformers sentencepiece psutil tqdm ipykernel jupyterlab spacy \
    ipywidgets torch
```

The notebook also records the actual runtime package versions in:

```text
requirements_runtime.txt
```

---

## Running the Notebook

1. Clone the repository.
2. Create and activate a Python 3.12 environment.
3. Install the required packages and the Spanish spaCy model.
4. Place the required research files inside the `data/` directory.
5. Open the notebook:

```bash
jupyter lab
```

6. Run `codigoBIAS_rs_CYG.ipynb` from top to bottom.

The notebook contains explicit stages for preparing the human-annotation spreadsheets. If the annotation files have not yet been completed, execution must pause at the annotation stage until the required human labels are available.

---

## Main Experimental Settings

The final experiments use the following main configuration:

| Setting | Value |
|---|---|
| Random state | 5 |
| Holdout split | 80/20 |
| Word2Vec dimensions | 50, 100, 200 |
| Word2Vec architecture | Skip-gram |
| Context window | 5 |
| Main expansion | Top-100 |
| Top-\(N\) sensitivity | 25, 50, 100, 200 |
| Candidate \(k\) values | 2–10 |
| Repeated cross-validation | 5 folds × 2 repetitions |
| HDBSCAN `min_cluster_size` | 20 |
| HDBSCAN `min_samples` | 5 |
| Independent human reference | 600 tweets |
| Silver-standard human audit | 250 tweets |

---

## Outputs

The notebook creates a consolidated experimental workbook:

```text
outputs_bias_aware/resultados_experimentos_revisores.xlsx
```

It also exports individual CSV files for the main experimental tables, including:

- dataset and preprocessing audits;
- holdout results;
- objective \(k\)-selection results;
- Top-\(N\) lexicon expansion;
- complete Top-100 expanded vocabularies;
- clustering comparisons;
- repeated cross-validation results;
- Wilcoxon/Holm statistical tests;
- silver-standard validation;
- human-annotation agreement;
- independent human validation;
- representation benchmarks;
- descriptive coverage analyses.

These outputs are intended to support traceability between the notebook and the tables, figures, appendices, and conclusions reported in the associated manuscript.

---

## How to Interpret the Results

A central design principle of this repository is that **coverage, pseudo-label fidelity, and human-validated correctness are different quantities**.

An expanded lexicon may identify more candidate political content without necessarily improving political-content classification. For this reason, the notebook evaluates expansion using independent human judgments and error-oriented metrics rather than treating increased coverage as automatic evidence of bias reduction.

In the 100-dimensional experiment reported in the manuscript, Top-100 expansion increased political-content coverage from **79.13% to 82.78%**, while the independent human evaluation showed unchanged recall and lower specificity under the expanded scenario. This illustrates the coverage–error trade-off that the framework is designed to make visible.

---

## Data Availability

The research data are not publicly distributed because they contain politically sensitive social-media content associated with public protests, social unrest, and demonstrations against the government. Public release of the raw data could expose user-generated content, metadata, or location-related information and may create risks of re-identification or misuse.

As stated in the manuscript:

> The data presented in this study are available on request from the corresponding author. The data are not publicly available due to privacy and ethical restrictions.

The source code and reproducibility workflow can therefore be shared independently of the restricted raw dataset.

---

## Associated Manuscript

**A Bias-Aware Framework for Political Content Detection in Social Media Using Semantic Lexicon Expansion**

**Authors**

- Cleopatra Guerra-Almeida
- Edison Loza-Aguirre
- Lorena Recalde
- Carlos Ayala-Tipan

The manuscript describes the methodological motivation, evaluation design, statistical analysis, limitations, and interpretation of the experiments implemented in this repository.

---

## Reproducibility Notes

This repository is designed to preserve the distinction between model development and external evaluation.

In particular:

- independent human labels are not used to build the semantic space;
- evaluation folds do not influence Word2Vec training or lexicon expansion;
- clustering is re-estimated inside the corresponding training partition;
- pseudo-labels are treated as weak supervisory signals rather than human ground truth;
- statistical comparisons are based on paired fold-level results;
- the expanded vocabulary remains directly inspectable.

These choices are intended to make the semantic-expansion process transparent and auditable.

---

## Citation

If you use this repository, please cite the associated manuscript once its final bibliographic information is available.

```text
Guerra-Almeida, C.; Loza-Aguirre, E.; Recalde, L.; Ayala-Tipan, C.
A Bias-Aware Framework for Political Content Detection in Social Media
Using Semantic Lexicon Expansion.
```