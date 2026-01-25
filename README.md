# 📈 TimeComplexity

This project analyzes the correlation between **Sampling Rate**, **Data Entropy**, and **Computational Complexity** using the **USAD** (UnSupervised Anomaly Detection) model.

---

## 🎯 Study Goals
- **Information Analysis**: Measure how much **Entropy** (information density) is preserved when downsampling time-series data.
- **Efficiency Optimization**: Find the optimal sampling rate that minimizes **Time Complexity** while maintaining high detection accuracy.
- **Real-time Trade-off**: Evaluate the balance between model inference speed and anomaly detection performance.

---

## 🔬 Core Experiments

### 1. Decimation & Sampling
- Systematically reducing the data resolution (Sampling Rate) to observe changes in data structure.
- Analyzing the loss of signal characteristics through the lens of information theory.

### 2. USAD Model Training
- Utilizing the **UnSupervised Anomaly Detection (USAD)** framework (Encoder-Decoder based Adversarial Training).
- Fine-tuning models across different decimation levels to benchmark performance.

### 3. Correlation Analysis
- Benchmarking **F1-Score** vs. **Processing Time**.
- Investigating how Latent Space features represent the complexity of the original signal.

---

## 🛠 Tech Stack
- **Deep Learning**: PyTorch
- **Algorithm**: USAD (anomaly detection model)
- **Data Analysis**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn

---

## 📂 Project Structure
- `usad-sampling-rate.ipynb`: Main experiment notebook containing data preprocessing, model training, and entropy-based analysis.

---
