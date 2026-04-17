# Heart Disease Prediction using Machine Learning

## Overview
This project builds a machine learning model to predict heart disease using clinical and engineered features.

It demonstrates a ML pipeline:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training & Evaluation
- Model Comparison
- Interpretability (SHAP)
- Final Model Selection

## Problem statement

Predict whether a patient has heart disease:

- 0 → No disease
- 1 → Disease

This is a **binary classification problem**.

## Dataset

- Source: [Kaggle Heart Disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- ~300 patient records
- Includes clinical variables such as:
    - Age
    - Chest pain type (cp)
    - Maximum heart rate (thalach)
    - Thalassemia (thal)

![correlation_matrix](./assets/correlation_matrix.png)


## Project Structure
├── data/
├── notebooks/
├── src/
├── models/
├── assests/
├── README.md


## Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib / Seaborn


## Exploratory Data Analysis

### Categorical attributes

![graphics_categorical_attributes](./assets/graphics_categorical_attributes.png)

### Continuous attributes

![graphics_continuous_attributes](./assets/graphics_continuous_attributes.png)


## Feature Engineering

est_stroke_volume : 

df['cardiac_capacity'] = df['thalach'] / df['age']

df['isquemia_score'] = (df['oldpeak'] / df['oldpeak'].max()) + df['exang'] + (df['ca'] / 3)

df['est_stroke_volume'] = (df['trestbps'] / df['thalach']) * (df['age'] / 50)

## Insight

Feature engineering improved performance from:  86% → 93% accuracy (+7%)

## Ablation Study

Feature Set	Accuracy
Base features	86%
Reduced engineered features	89%
Full feature set	93%

## Models Evaluated
- Logistic Regression
- Random Forest
- XGBoost

## Model Performance

Model	AUC
Logistic Regression	0.91
XGBoost	0.98
Random Forest	0.999

## Cross-Validation (5-fold)

Model	CV AUC	Std
Logistic Regression	0.906	±0.034
XGBoost	0.981	±0.012
Random Forest	0.9987	±0.0025

Random Forest achieved the highest performance with extremely low variance, indicating strong generalization.

## Model Interpretability (SHAP)

SHAP was used to analyze feature contributions.

Findings:
No single feature dominates the model
Top feature contributes ~29% of total importance
Model relies on multiple complementary signals


![SHAP (test set)](./assets/SHAP (test set).png)


# Confusion Matrix

[[91  9]
 [ 5 100]]
False Negatives: 5
False Positives: 9

The model achieves 95% recall for disease detection, minimizing missed cases — critical in healthcare.

![Confusion matrix_ XGBoost (test set)](./assets/Confusion matrix_ XGBoost (test set).png)


# ROC Curve
 
ROC-AUC ≈ 0.98–0.999
Curve close to top-left corner

The model demonstrates excellent class separability and robustness across thresholds.

![ROC_curve_XGBoost (test set)](./assets/ROC_curve_XGBoost (test set).png)

## Final Model
Random Forest
- Highest AUC
- Lowest variance
- Strong generalization
- Simpler than XGBoost

## Key Takeaways

Feature engineering can significantly improve performance
Cross-validation is essential for model validation
SHAP helps avoid misleading feature importance interpretations
Model selection should consider both performance and stability

## Conclusion

This project demonstrates a complete machine learning workflow, from data exploration to model selection, highlighting the impact of feature engineering and proper validation techniques.


## Author

Jhoselyn Miluska Pajuelo
Software Engineer | ML Enthusiast