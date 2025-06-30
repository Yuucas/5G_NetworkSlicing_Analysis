# 5G Network SLA Breach Prediction

## 1. Project Overview

This project moves beyond simple network traffic classification and focuses on a critical, high-value task for telecommunications operators: **proactively predicting Service Level Agreement (SLA) breaches** in a 5G network.

Instead of reacting to network failures after they have impacted customer experience, this model acts as an early warning system. By analyzing a session's characteristics at its inception, the model forecasts whether it is likely to violate its Quality of Service (QoS) guarantees.

This repository contains the full Python code for data analysis, feature engineering, model training, and evaluation, using a public dataset on 5G network slicing.

## 3. Methodology

The project follows a structured workflow from raw data to an actionable predictive model.

#### a. Dataset
The analysis is based on the "Network Slicing in 5G" dataset, which contains records of various user sessions with their associated network parameters.

Kaggle Dataset Link: [https://www.kaggle.com/datasets/amohankumar/network-slicing-in-5g/data](https://www.kaggle.com/datasets/amohankumar/network-slicing-in-5g/data)

#### b. Intelligent Feature Engineering
To provide the model with deep domain knowledge, several features were engineered to capture telecommunications logic:

| Feature Name          | Type      | Rationale                                                                                                              |
| :-------------------- | :-------- | :----------------------------------------------------------------------------------------------------------------------- |
| `QoS_Violation_Score` | Numerical | Creates a score based on whether the `Packet delay` and `Packet Loss Rate` are acceptable for a given service type.      |
| `Session_Intensity`   | Numerical | Combines device capability (`LTE/5g Category`) and `GBR` status to quantify the resource strain a session places on the network. |
| `Is_Critical_Service` | Binary    | A flag that aggregates high-priority use cases (`Healthcare`, `Industry 4.0`, etc.) into a single "must-not-fail" signal. |

#### c. Target Variable: `Will_Violate_SLA`
The core of this forecasting task is the target variable. It was created directly from our engineered `QoS_Violation_Score`:
- If `QoS_Violation_Score > 0`, the session is labeled as **1 (`SLA Violated`)**.
- If `QoS_Violation_Score == 0`, the session is labeled as **0 (`SLA Not Violated`)**.

#### d. Model Training and Evaluation
A **Random Forest Classifier** was chosen for its robustness and interpretability. The model was trained on the preprocessed data and its performance was evaluated using a **Confusion Matrix**.