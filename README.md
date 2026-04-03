# RNNs-weak-supervision-anomaly-detection


# Candy Defect Detection & Root Cause Analysis Using RNNs

This project focuses on detecting and analyzing defects in a candy production process using time-series data collected from sensors across the manufacturing line.

## Project Goals

1. **Predictive Modeling**
   Build a recurrent neural network (RNN)-based model that predicts whether a given candy has any defects and, if so, identifies the types of defects.

2. **Root Cause Analysis**
   Identify patterns in the sensor data that are associated with defects. This will help the manufacturer understand which process anomalies lead to specific defects.


## Dataset

* Sensor-based time series capturing the candy manufacturing process.
* Each candy may have zero, one, or multiple defects - multi label scenario.
* Defects are associated with patterns of varying lengths across the time series.


## Results and Methods

### version 1:

* **Model Architecture**

  * Bidirectional GRU with a linear temporal head
  * Outputs per-timestep probabilities for each defect
  * Aggregation across timesteps for final candy-level predictions

* **Preprocessing & Dataset Handling**

  * Padding of variable-length sequences using PyTorch utilities
  * Custom `Dataset` and `collate_fn` for batching

* **Defect Explanation / Root Cause Analysis**

  * Visualizes temporal probabilities over the sequence
  * Highlights intervals in the time series responsible for each defect
  * Can detect defects spanning multiple sensors (including simultaneous patterns)

* **Evaluation Metrics**

  * Frame-level and candy-level accuracy and F1-score
  * Demonstrates macro-F1 ~0.69
  * Visualization provides intuitive root cause analysis

* **Flexibility**

  * Approach generalizes to other datasets after retraining
  * Not dependent on known synthetic patterns (sinusoids)

<img width="1441" height="834" alt="image" src="https://github.com/user-attachments/assets/47562eff-1486-42cf-9b6f-e9d06b035df7" />


### version 2
* the 2nd version focus on improving perfomace on defect that previously had low accuracy
  

### 2. Data Augmentation

* Random cropping during training improves model robustness to variable-length sequences.
* Sliding window with stride is used at inference to create overlapping segments, allowing finer temporal localization of defects.

### 3. Model Architecture

* Bidirectional GRU (hidden_dim=64) to capture temporal dependencies.
* Temporal head: linear layer with sigmoid for multi-label anomaly detection per timestep.
* Aggregation of temporal predictions:

  * Sequence-level: `max` pooling across time.
  * Frame-level: average predictions per timestep from overlapping windows.
  
## raining and Inference

* Optimizer: Adam (`lr=1e-3`)
* Loss: Binary Cross-Entropy (BCE)
* Training uses random crop augmentation for each sequence.
* Inference uses sliding windows with stride to aggregate predictions.

## Results

### Sequence-level Performance (Test Set)

| Metric     | Score |
| ---------- | ----- |
| F1 (macro) | 0.980 |
| Accuracy   | 0.951 |

### Frame-level Performance (Windowed Aggregation)

| Defect       | Accuracy | F1    |
| ------------ | -------- | ----- |
| Defect 0     | 0.974    | 0.269 |
| Defect 1     | 0.984    | 0.378 |
| Defect 2     | 0.989    | 0.742 |
| Defect 3     | 0.972    | 0.757 |
| Defect 4     | 0.976    | 0.729 |
| Macro F1     | —        | 0.575 |

 Frame-level F1 is lower for rare or very short defects due to stride and thresholding.

<img width="1408" height="822" alt="image" src="https://github.com/user-attachments/assets/e84b6752-84a4-4ec7-9db4-50537390e693" />

