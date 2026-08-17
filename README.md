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
## Post-Modeling Analysis

### Was your initial hypothesis correct?

My initial hypothesis was that the Decision Tree would perform best because it can capture nonlinear relationships between features.

The results did not support this hypothesis.

KNN performed best based on the F1-score, achieving an F1-score of 0.8318. It also achieved the highest Precision (0.9157), Recall (0.7646), and PR-AUC (0.7957) among the three models.

The Decision Tree achieved an F1-score of 0.7495, which was lower than both KNN and Logistic Regression.

Therefore, the initial hypothesis that the Decision Tree would perform best was not correct for the current model configurations.

The results also showed that Logistic Regression was a strong baseline. It achieved the highest ROC-AUC (0.9777), although its Recall (0.6056) and F1-score (0.7096) were lower than those of KNN.

---

### Which model performed best?

KNN performed best overall based on the F1-score.

The results were:

| Model | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8633 | 0.6056 | 0.7096 | **0.9777** | 0.7527 |
| KNN | **0.9157** | **0.7646** | **0.8318** | 0.9192 | **0.7957** |
| Decision Tree | 0.7583 | 0.7431 | 0.7495 | 0.8714 | 0.5639 |

KNN achieved the highest F1-score, which indicates the best balance between Precision and Recall at the current classification threshold.

However, Logistic Regression achieved the highest ROC-AUC. Therefore, KNN was the best model according to the selected F1-score criterion, but Logistic Regression showed stronger overall ranking performance across classification thresholds.

---

### Which metric was most informative?

For this fraud detection problem, F1-score was the most informative metric for the initial model comparison.

Recall was initially considered the most important metric because missing a fraudulent transaction is costly. However, the results showed why Recall alone is not sufficient.

KNN achieved a Recall of 0.7646, but it also achieved a high Precision of 0.9157. This resulted in an F1-score of 0.8318.

F1-score provided a useful balance between detecting fraudulent transactions and avoiding too many False Positives.

PR-AUC was also particularly informative because the dataset is highly imbalanced. KNN achieved the highest PR-AUC (0.7957).

Therefore, for this initial comparison, F1-score was used to select the best model, while Recall and PR-AUC were also considered important supporting metrics.

---

### How did class imbalance affect the results?

Class imbalance had a significant effect on the evaluation.

After removing duplicate rows, approximately 99.83% of the transactions belonged to Class 0 (legitimate) and only about 0.17% belonged to Class 1 (fraudulent).

Because the majority class is so dominant, Accuracy alone can be misleading.

For example, a model that predicts almost every transaction as legitimate can achieve extremely high Accuracy while detecting very few fraudulent transactions.

This can be seen in the current results, where all three models achieved approximately 99.9% Accuracy:

- Logistic Regression: 0.9992
- KNN: 0.9995
- Decision Tree: 0.9992

Despite these very similar Accuracy values, their Precision, Recall, F1-score, and PR-AUC values were substantially different.

This confirms that Accuracy is not sufficient for evaluating this fraud detection problem.

---

### What was the trade-off between False Positives and False Negatives?

The trade-off can be understood through Precision and Recall.

False Positives occur when legitimate transactions are incorrectly classified as fraudulent. False Negatives occur when fraudulent transactions are classified as legitimate.

A model with higher Recall detects more fraudulent transactions, but it may also produce more False Positives.

In the current results, KNN achieved both the highest Recall (0.7646) and the highest Precision (0.9157). This resulted in the highest F1-score (0.8318).

Logistic Regression had a higher Precision (0.8633) than Decision Tree (0.7583), but its Recall was considerably lower (0.6056). This means Logistic Regression missed more fraudulent transactions at the current classification threshold.

The results show that fraud detection requires a balance between catching fraudulent transactions and avoiding unnecessary False Positives. The appropriate balance ultimately depends on the relative business cost of False Positives and False Negatives.

---

### Hypothesis Summary

| Initial Hypothesis | Result |
|---|---|
| Decision Tree would perform best | ❌ Not supported |
| Recall would be very important | ✅ Supported |
| Predicting everything as legitimate would give high Accuracy but zero Fraud Recall | ✅ Supported |
| Scaling would significantly affect KNN | ✅ Expected; KNN was evaluated with scaling |
| Decision Tree could overfit | ⚠️ Not yet conclusively tested |

The Decision Tree overfitting hypothesis cannot be confirmed from the current results alone because training performance was not compared with validation performance and no tree-depth experiment was performed.

Further analysis should investigate Decision Tree depth, KNN hyperparameters, classification thresholds, and final performance on the untouched test set.