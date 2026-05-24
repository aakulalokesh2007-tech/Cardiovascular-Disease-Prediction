# Cardiovascular Disease Prediction ❤️📊

## Overview
This project is a Machine Learning pipeline designed to predict the likelihood of a patient having cardiovascular disease based on their medical records and physical attributes. 

## Dataset
The dataset contains 70,000 records of patient data. After rigorous data pre-processing and outlier removal, the final dataset consisted of 68,408 highly reliable entries. Features include age, height, weight, blood pressure (systolic and diastolic), cholesterol, and lifestyle habits.

## Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (SVM, Logistic Regression, Decision Tree, Random Forest, KNN)

## Key Findings
* **Age Factor:** The likelihood of cardiovascular disease significantly increases as patients cross the age of 50.
* **Blood Pressure:** Systolic (ap_hi) and Diastolic (ap_lo) blood pressures showed the strongest correlation with the target variable.

## Model Performance
Five different algorithms were trained and evaluated. The **Support Vector Machine (SVM)** yielded the highest accuracy.

* **Support Vector Machine (SVM):** 73.67%
* **Logistic Regression:** 72.91%
* **Decision Tree:** 72.88%
* **Random Forest:** 70.27%
* **K-Nearest Neighbor (KNN):** 69.87%

![Model Accuracies](model_accuracies.png) 
*(Note: Ensure your image file is actually named this in your repo!)*
