# COPD Data Mining

## Project Overview
This repository contains a data mining study on **Chronic Obstructive Pulmonary Disease (COPD)** patients. The study involves **data preprocessing, missing value imputation, outlier detection, Principal Component Analysis (PCA), regression, and machine learning models** to analyze and predict COPD-related factors.

## Features
- **Dataset Analysis**: Understanding key features and their impact on COPD.
- **Data Preprocessing**:
  - Handling missing values using machine learning models (MARS & Random Forest).
  - Standardization and normalization of variables.
  - Outlier detection and treatment.
  - Principal Component Analysis (PCA) for dimensionality reduction.
- **Machine Learning Models Used**:
  - Multivariate Adaptive Regression Splines (MARS)
  - Random Forest (RF)
  - Logistic Regression (LR)
  - Decision Trees (DT)
  - Gradient Boosting Machines (GBM)
  - Support Vector Machines (SVM)
  - k-Nearest Neighbors (k-NN)
  - XGBoost (Extreme Gradient Boosting)
- **Performance Evaluation Metrics**:
  - Accuracy, RMSE, R², ROC-AUC, Mean Absolute Error (MAE), and other statistical measures.

## Dataset
The dataset consists of demographic details, lifestyle habits, medical conditions, and pulmonary function test results of COPD patients. The primary features include:

- **Demographic Information**: Age, gender, education level, occupation.
- **Lifestyle Habits**: Smoking status, duration, frequency.
- **Medical Conditions**: Family history of COPD/asthma, pulmonary function test results (FEV1, PEF, FEV1/FVC ratio).

## Installation & Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/COPD_Data_Mining.git
   ```
2. Navigate to the project folder:
   ```sh
   cd COPD_Data_Mining
   ```
3. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the main script:
   ```sh
   python main.py
   ```

## Results
- **Best Performing Model:** XGBoost with **92% accuracy** and **0.95 ROC-AUC**.
- **Findings:** High correlation between smoking history and COPD progression. MARS model performed best for missing value imputation.


