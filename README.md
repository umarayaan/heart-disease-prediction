# Heart Disease Prediction

Predicts whether a patient has heart disease from 13 clinical features, using scikit-learn. The project covers data checks, EDA, a leakage-safe preprocessing pipeline, a six-model comparison, and hyperparameter tuning.

## Dataset
- 180 patients, 13 clinical features plus a patient ID
- `values.csv`: patient features
- `labels.csv`: target `heart_disease_present` (0 = healthy, 1 = heart disease)
- Classes: 100 healthy, 80 with heart disease (55.6% / 44.4%)

## Approach
1. **Data preparation:** merged features and labels on `patient_id`, dropped the ID, checked for missing values, duplicates and impossible values (none found).
2. **EDA:** univariate and bivariate analysis on the training set only.
3. **Split:** stratified 80/20 train/test split.
4. **Preprocessing:** `ColumnTransformer` inside a `Pipeline` (scaling for numeric columns, one-hot encoding for categorical columns, binary columns passed through). Tree models use encoding only.
5. **Model comparison:** repeated stratified 5-fold cross-validation (3 repeats) scored on ROC-AUC, recall, F1 and accuracy.
6. **Tuning:** `GridSearchCV` on the top three models.

## Model comparison (cross-validation, before tuning)

| Model | AUC | Recall | F1 | Accuracy |
|-------|-----|--------|----|----------|
| Random Forest | 0.861 | 0.735 | 0.749 | 0.778 |
| Logistic Regression | 0.859 | 0.755 | 0.777 | 0.806 |
| SVM | 0.853 | 0.723 | 0.740 | 0.778 |
| Gradient Boosting | 0.830 | 0.703 | 0.706 | 0.745 |
| KNN | 0.810 | 0.650 | 0.687 | 0.743 |
| Decision Tree | 0.735 | 0.678 | 0.670 | 0.713 |

After tuning, cross-validated ROC-AUC was 0.884 for Random Forest, 0.882 for SVM and 0.878 for Logistic Regression. The three are statistically tied, so **Logistic Regression** was chosen as the final model because it is the simplest and most interpretable.

## Key findings
- Strongest predictors: `thal` (reversible defect), `chest_pain_type` (type 4), `num_major_vessels`, `max_heart_rate_achieved`, `oldpeak` and `sex`.
- Cholesterol and resting blood pressure have almost no linear relationship with the label on their own.

## Limitations
- Only 180 rows, so scores are noisy (the test set has 36 patients).
- Recall is around 0.75, so roughly 1 in 4 sick patients is missed. Decision-threshold tuning was not done.
- This is a learning project, not a clinical tool.


## Files
- `heart_disease_prediction.ipynb`: full analysis, models and report on challenges faced
- `values.csv`, `labels.csv`: dataset


## Tech stack
Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter
