# Intelligent Clinical Early Warning System 🏥🚨

**Author**: Muhammad Hammad (Roll No. 22F-BSAI-39)
**Course**: Deep Learning - Complex Computing Problem

## 📌 Project Overview
Every year, thousands of patients deteriorate silently in hospitals before warning signs are noticed[cite: 3]. This project builds a multi-generational Artificial Intelligence Early Warning System to monitor patients continuously and flag those at risk of critical events (e.g., sepsis, cardiac arrest, or ICU transfer)[cite: 3]. 

The deep learning pipeline is designed to predict patient deterioration risk using a combination of tabular vital signs and free-text clinical notes[cite: 3]. Because a missed deterioration (false negative) can be fatal while a false alarm is merely an inconvenience, this system heavily prioritizes **Recall** over standard accuracy, utilizing weighted loss functions to penalize missed detections[cite: 2, 3].

## 📊 Dataset & Preprocessing
This project utilizes the **Kaggle Patient Survival Prediction** dataset[cite: 1, 3]. 
* **Imbalance Handling**: The dataset features a heavy class imbalance, with only about 8-10% of patients experiencing hospital death/deterioration[cite: 2]. A `pos_weight` of ~10.58 is applied to the Binary Cross-Entropy loss to heavily penalize false negatives[cite: 1, 2].
* **Tabular Preprocessing**: Missing numerical values are imputed using the median, and categorical columns use the mode. Categorical features are one-hot encoded[cite: 1].
* **Synthetic Clinical Notes**: Since the dataset is strictly tabular, synthetic clinical notes are dynamically generated from physiological measurements (e.g., translating a high heart rate into "tachycardic" and low blood pressure into "hypotensive") to train the NLP models[cite: 1, 2].

## 🧠 Model Generations

The project tracks the evolution of deep learning architectures across three generations:

### Generation 1: Baseline DNN
A two-hidden-layer feedforward network establishing the performance floor[cite: 2]. 
* **Optimizers**: Compares SGD (with momentum) against Adam. Adam converged faster and smoother early on due to its adaptive per-parameter learning rates[cite: 1, 2].
* **Regularization**: Evaluates the ablation of Dropout and Batch Normalization. Dropout proved essential for maintaining Recall and preventing the network from simply memorizing the majority "stable" class[cite: 1, 2].

### Generation 2: Capturing the Patient Timeline (RNNs)
Evaluates recurrent architectures (LSTM, GRU, and Bi-LSTM) to capture temporal dependencies in patient history[cite: 3].
* **LSTM vs. GRU**: LSTMs use three gates to solve the vanishing gradient problem over long sequences. The GRU simplifies this to two gates, resulting in faster training times with competitive Recall[cite: 2, 3].
* **Real-time vs. Retrospective**: While Bi-LSTMs achieved higher offline F1 scores by reading sequences backwards and forwards, they leak "future" data at inference time[cite: 2]. Therefore, unidirectional GRUs/LSTMs are recommended for live monitoring, while Bi-LSTMs are strictly for retrospective audits[cite: 2, 3].

### Generation 3: Reading Clinical Notes (Transformers)
Fine-tunes the `emilyalsentzer/Bio_ClinicalBERT` pre-trained model on the synthetic clinical notes[cite: 1, 3]. 
* **Frozen vs. Full Fine-Tuning**: 
  * *Strategy A (Frozen)*: Trains only the 2-layer classification head (~1,500 parameters). Fast and memory-light (approx. 179.8s)[cite: 1, 2].
  * *Strategy B (Full Fine-Tune)*: Updates all ~110 million weights. Computationally expensive (approx. 884.0s) but adapts clinical language representations to the task, yielding higher Recall[cite: 1, 2].
* **Self-Attention**: Visualizing the final-layer attention weights reveals that the model appropriately focuses on high-salience clinical terms like "hypotensive" and "tachycardic" rather than proximity or word order[cite: 1, 2].

## 🏆 Unified Model Comparison

Below is the evaluation on the 20% test split, showcasing Accuracy, Precision, Recall, F1-Score, and Training Time[cite: 1, 2]:

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DNN (Baseline)** | 0.815 | 0.279 | 0.727 | 0.404 | ~154.0s |
| **LSTM** | 0.886 | 0.372 | 0.472 | 0.416 | ~48.6s |
| **Bi-LSTM** | 0.896 | 0.397 | 0.398 | 0.397 | ~56.6s |
| **GRU** | 0.875 | 0.344 | 0.499 | 0.407 | ~46.7s |
| **ClinicalBERT (Frozen)** | 0.805 | 0.195 | 0.402 | 0.263 | ~179.8s |
| **ClinicalBERT (Full Fine-tune)** | 0.791 | 0.196 | 0.460 | 0.275 | ~884.0s |

*Note: The GRU achieves the best speed-to-recall trade-off among recurrent models, while the baseline DNN achieved high recall due to hyperparameter tuning tailored to the class imbalance.*[cite: 2]

## 🚀 How to Run

1. **Prerequisites**: Ensure you have Python installed along with `pandas`, `numpy`, `torch`, `scikit-learn`, `matplotlib`, `seaborn`, and `transformers`[cite: 1].
2. **Kaggle API Key**: You must provide a valid `kaggle.json` file. The notebook attempts to load this from Google Colab Secrets (`KAGGLE_USERNAME` and `KAGGLE_KEY`) or via a manual file upload[cite: 1].
3. **Dataset Download**: The script automatically provisions the `patient-survival-prediction` dataset from Kaggle and extracts `train.csv` into a `data_kaggle/` directory[cite: 1].
4. **Execution**: Run the Jupyter Notebook cells sequentially to reproduce preprocessing, model training, ablations, and attention heatmap generations.
