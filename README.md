# mini-project-01

## Project Overview

This project focuses on building a machine learning pipeline for detecting fraudulent transactions.

## Models

The following classification models will be implemented and compared:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree

## Hypothesis Before Modeling

### 1. Which model do you expect to perform best for fraud detection? Why?

I expect the Decision Tree to perform best because it can capture nonlinear relationships between features. However, Logistic Regression may provide a strong baseline because this is a binary classification problem.

### 2. Which metric is more important for this problem: Precision, Recall, or F1-score? Why?

I think Recall is more important because we want to detect as many fraudulent transactions as possible. However, Precision and F1-score are also important because too many False Positives can be a problem.

### 3. What do you expect to happen if the model predicts all transactions as legitimate?

I expect the model to have very high Accuracy because most transactions are legitimate. However, it will fail to detect fraudulent transactions, so the Recall for the Fraud class will be zero.

### 4. Do you expect feature scaling to significantly affect KNN performance?

Yes. I expect feature scaling to significantly affect KNN because KNN uses distances between data points. Features with larger ranges can otherwise dominate the distance calculation.

### 5. Do you expect the Decision Tree to overfit? Why?

Yes. I expect the Decision Tree to overfit, especially if the tree becomes too deep. A deep tree can learn noise and specific patterns from the training data instead of general patterns.

# After Training Analysis

### Was your initial hypothesis correct?

TBD — I will compare the actual results with my initial hypotheses.

### Which model performed best?

TBD — I will determine this based on the evaluation results.

### Which metric was most informative?

TBD — I will compare Precision, Recall, and F1-score, with particular attention to the Fraud class.

### How did class imbalance affect the results?

Class imbalance can make Accuracy misleading because a model can achieve high Accuracy by predicting most transactions as legitimate while failing to detect fraud.

### What was the trade-off between False Positives and False Negatives?

False Positives mean legitimate transactions were classified as fraud, while False Negatives mean fraudulent transactions were classified as legitimate. In fraud detection, False Negatives are particularly important because they represent missed fraud.



