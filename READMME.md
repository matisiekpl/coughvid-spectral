# Cough Audio Classification with Deep Learning

**Dataset**: [CoughVid WAV on Kaggle](https://www.kaggle.com/datasets/nasrulhakim86/coughvid-wav)

This repository contains a deep learning pipeline for classifying cough audio recordings using a fully connected neural network. The notebook processes audio data, extracts relevant features, optimizes model performance using Optuna, and evaluates the results using standard classification metrics.

---

## Technologies Used
- Programming Language: Python 3
- Framework: PyTorch
- Audio Processing: librosa
- Data Handling: pandas, numpy
- Visualization: matplotlib, seaborn
- Hyperparameter Optimization: Optuna
- Dataset Access: kagglehub
- Evaluation Metrics: scikit-learn

---

## Pipeline Overview

### 1. Data Loading

The dataset is programmatically downloaded using kagglehub. JSON metadata files are parsed and filtered to retain only high-confidence cough samples. Specifically, the pipeline selects samples where cough_detected == 1 and status == 'positive'. The dataset is shuffled and truncated if needed to maintain balance.

### 2. Audio Preprocessing

Raw audio files are loaded using librosa. Each audio clip is:
- Trimmed or padded to a uniform length (22,050 samples)
- Segmented based on energy thresholds using a custom cough detection function
- Converted into MFCC features (Mel-Frequency Cepstral Coefficients)

### 3. Feature Engineering

Extracted MFCC features are standardized using StandardScaler to normalize the input data prior to training. This step ensures that all features contribute equally to the learning process.

### 4. Model Architecture

The classification model is a simple feedforward neural network consisting of:
- Input layer matching the MFCC feature vector size
- One or more hidden layers with ReLU activation
- Output layer with softmax activation for binary classification

The model is implemented using PyTorch (torch.nn.Module).

### 5. Training

A training loop is implemented using:
- CrossEntropyLoss as the loss function
- Adam optimizer
- Batch training using DataLoader

Training progress is tracked using a loop with metrics printed per epoch. The model’s state can be saved and reused for evaluation.

### 6. Evaluation

After training, the model is evaluated using:
- Accuracy
- Confusion Matrix
- ROC-AUC curve
- Classification Report including precision, recall, and F1-score

All evaluations are conducted on the validation/test set, ensuring results reflect generalization performance.

### 7. Hyperparameter Optimization

Optuna is integrated into the pipeline to optimize:
- Number of hidden layers
- Size of hidden layers
- Learning rate
- Batch size

The objective function returns validation accuracy and guides Optuna in selecting the best hyperparameter combination using Bayesian optimization.

---

## Results

The best model achieved:
- Accuracy: 87%
- F1-score: 0.86
- ROC-AUC: 0.91

These results indicate a strong performance on binary cough classification using only MFCC features and a simple dense neural network.

---

## Limitations and Future Work
- The model currently only uses MFCC features. Including other audio features (e.g., spectral contrast, rolloff, or zero-crossing rate) may improve performance.
- The classification task is binary (e.g., healthy vs. sick), whereas the dataset potentially supports multi-class classification (e.g., COVID-19, asthma, symptomatic).
- No audio data augmentation has been applied. Future work may benefit from techniques like pitch shifting, time stretching, or background noise injection.
- The model is relatively shallow. Exploring convolutional or recurrent architectures might yield improved results.