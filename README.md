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

## Output

After running the notebook, you will obtain several files and visualizations that summarize the performance and bias analysis of the various models. These include:

- **CSV Files:**  
  - `model_performance_metrics.csv`: Contains evaluation metrics (accuracy, precision, recall, F1-score) for each model and configuration.  
  - Prediction result files (e.g., `naive_bayes_with_tfidf_results.csv`, `kNN_with_BOW_results.csv`, `bert_predictions.csv`, etc.) for detailed analysis of model outputs.  
  - `bias_test_metrics.csv`: Summarizes metrics from tests on bias-modified datasets.

- **Visualization Outputs:**  
  - Bar charts displaying the most frequent unigrams and bigrams in the preprocessed abstracts.  
  - Confusion matrices and ROC curves for evaluating model performance.  
  - Word clouds and n-gram frequency charts as part of error analysis.  
  - Line graphs comparing performance across the original test set and various bias-tested datasets (e.g., Gender Bias, Age Bias).

## Contributing

Contributions and improvements to this project are welcome! If you find a bug, have suggestions, or would like to contribute new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Datasets:** The biomedical datasets used in this project are sourced from [Gaborandi’s datasets on Hugging Face](https://huggingface.co/Gaborandi).
- **Tools:** Special thanks to the open-source community for providing robust libraries such as `datasets`, `nltk`, `scikit-learn`, `matplotlib`, `wordcloud`, `transformers`, and `torch`.

Happy coding!

