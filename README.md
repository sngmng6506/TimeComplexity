# TimeComplexity (Study Project)

This repository is a **personal study project** focused on understanding  
**time complexity, sampling rate, and their effects on time-series data processing**.

The goal of this project is not to build a production-ready system,  
but to explore how data size, sampling rate, and algorithmic choices  
impact computational cost and data representation.

---

## 🎯 Study Goals

- Understand the concept of **time complexity** in practice
- Analyze how **sampling rate** affects time-series data size and structure
- Observe how changes in sampling rate influence:
  - Data resolution
  - Computational cost
  - Practical trade-offs in time-series processing
- Gain hands-on experience through Jupyter Notebook experiments

---

## 📘 Key Concepts

### ⏱ Time Complexity
Time complexity describes how the execution time of an algorithm grows  
as the size of input data increases (e.g., O(n), O(n²), O(log n)).

In time-series and signal processing tasks, time complexity is often affected by:
- Length of the sequence
- Sampling rate
- Model or algorithm design

---

### 🎚 Sampling Rate
Sampling rate refers to how frequently data points are collected from a signal or time series.

- Higher sampling rate  
  → More detailed data  
  → Larger data size and higher computational cost

- Lower sampling rate  
  → Reduced data size  
  → Possible loss of information

Choosing an appropriate sampling rate is a key trade-off between **performance and accuracy**.

---

## 📊 Notebook Overview

### 📌 `usad-sampling-rate.ipynb`

This notebook explores the relationship between **sampling rate and data processing cost**,  
likely in the context of **time-series or anomaly detection experiments**.

#### What this notebook covers:
- Loading or generating time-series data
- Applying different sampling rates
- Visualizing the impact of sampling rate changes
- Observing how data size and processing behavior change accordingly

This notebook serves as an experimental playground to better understand  
how preprocessing choices affect downstream computation.

---

## 🛠 Tech Stack

- Python
- Jupyter Notebook
- NumPy
- Matplotlib
- (Additional libraries depending on experiments)

---

## 📂 Project Structure

```text
TimeComplexity/
├── usad-sampling-rate.ipynb   # Sampling rate experiment notebook

