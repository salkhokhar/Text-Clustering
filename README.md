# From Data to Decisions: Tackling AI’s Hidden Bias

This repository contains the code for our group assignment (GNG_5125_Group_Assignment_2), which demonstrates a complete pipeline for text classification and analysis using biomedical abstracts. The notebook covers every step—from downloading and preprocessing datasets to feature engineering, model training (with various techniques including Bag-of-Words, TF-IDF, LDA, Word Embeddings, Naïve Bayes, k-Nearest Neighbors, Random Forest, and BERT), error analysis, and model bias evaluation.

## Repository Structure

- **GNG_5125_Group_Assignment_2.ipynb**  
  The main Jupyter Notebook containing the complete script and all analysis steps.

- **Output Files** (generated during execution):
  - `model_performance_metrics.csv` – CSV file with evaluation metrics for various models.
  - `bert_confusion_matrix.png` – Image file for the BERT model’s confusion matrix.
  - `bert_roc_curve.png` – ROC curve image for the BERT model.
  - Various CSV files with prediction results (e.g., `naive_bayes_with_tfidf_results.csv`, `kNN_with_BOW_results.csv`).
  - `bias_test_performance.png` – Graph showing model performance across bias-tested datasets.
  - `bias_test_metrics.csv` – CSV file summarizing metrics from bias evaluation.
  - `bert_predictions.csv` – Predictions from BERT evaluations.

## Installation

To run this notebook, you need Python 3 and the following libraries. You can install them via pip:

```bash
pip install datasets nltk pandas scikit-learn matplotlib wordcloud transformers torch
```
