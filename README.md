<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 20px; height: 55px">

# DSI-SG-42 Project 4

# In A Heartbeat: Prediction of Heart Disease Risk for Early Detection

---

## Introduction of Problem Statement

Heart disease remains the leading cause of death in the US, a statistic that has persisted for over a century.  Many of the contributing factors are controllable and detectable through early intervention. At Teladoc, a medical teleconsulting company, we are committed to empowering patients with knowledge and tools for proactive health management.

This project focuses on developing a heart disease risk prediction model. Our goal is to raise awareness of this prevalent condition and encourage early detection, which plays a crucial role in risk reduction.  This model will be presented to our company leadership for consideration as a valuable addition to our teleconsulting platform.

## Key Question

#### *How can we develop and integrate a data-driven feature that provides Teladoc end-users with a rigorous prediction of their risk for heart disease to facilitate early detection?*

## Table of Contents

Our data cleaning and modeling process are detailed in the following notebooks:  
  
[1. Import of Data](code/01_Data_Import_and_Cleaning.ipynb)  
[2A. Exploratory Data Analysis - Data Visualizations](code/02A_EDA_and_Data_Visualization.ipynb)  
[2B. Exploratory Data Analysis - Analysis on Missing Values](code/02B_EDA_MissingValues.ipynb)  
[2C. Exploratory Data Analysis - Before and After Imputation](code/02C_EDA_Before_and_After_Imputation.ipynb)  
[3. Supervised Learning](code/03_Modeling.ipynb)  

## Executive Summary

In recognizing the critical impact of heart disease - the leading cause of death in the US - and the significant potential for early detection to reduce risks, our mission is to leverage our teleconsulting platform to provide patients with accessible tools for early intervention. We embarked on creating a predictive model for heart disease risk, focusing on actionable insights for patient health management.  

In developing this model, our data process encompassed variable selection, outlier management, value mapping, and missing data imputation, leading to a robust modeling framework. The resulting Keras neural network model demonstrated superior performance in sensitivity and ROC AUC scores compared to a hypertuned Logistic Regression model, suggesting its potential as a more effective tool for early risk prediction.  

However, the Keras model’s complexity may impede real-time app analysis, necessitating regular updates to maintain efficacy. Additionally, expanding the data scope to include dietary habits, family history, and comprehensive health statistics could enhance predictive accuracy.  

We recommend integrating the model with resources for users identified as high-risk and ensuring the app interface remains user-friendly. Ongoing model reevaluation and data enrichment will be pivotal in refining this vital health management feature for our users.  

## Data Process

Our project utilized data from the [2022 Behavioral Risk Factor Surveillance System (BRFSS)]([text](https://www.cdc.gov/brfss/annual_data/annual_2022.html)), a public health survey conducted annually by the Centers for Disease Control and Prevention (CDC). This survey gathers information on health behaviors, chronic diseases, and preventive health practices from a representative sample of the US population. We converted the file from `.xpt` to `.csv` for easy handling in Python. The cleaning steps involved are:
1. Selection and renaming of relevant variables
2. Identify and handle outliers
3. Map numerical responses to corresponding text labels
4. Impute missing values

## Data Dictonary

The data dictionary displays the key columns used for this project:

| Column                    | Data Type | Dataset | Description                                                                                                      |
|---------------------------|-----------|---------|------------------------------------------------------------------------------------------------------------------|
| age                       | float64   | modeling_dataset.csv  | Age of respondent in years                                                                                      |
| height                    | float64   | modeling_dataset.csv  | Height of respondent in metres                                                                                  |
| weight                    | float64   | modeling_dataset.csv  | Weight of respondent in kg                                                                                      |
| bmi                       | float64   | modeling_dataset.csv  | Body Mass Index of respondent                                                                                   |
| yrssmok                   | float64   | modeling_dataset.csv  | Number of years the respondent has smoked                                                                       |
| packday                   | float64   | modeling_dataset.csv  | Number of cigarette packs the respondent smoked in a day                                                        |
| sleep_hours               | float64   | modeling_dataset.csv  | Number of hours sleep in a day                                                                                  |
| chd_mi                    | object    | modeling_dataset.csv  | Diagnosis of heart disease or myocardial infarction                                                             |
| health_status             | object    | modeling_dataset.csv  | Self-rated physical health: 'Excellent', 'Very Good', 'Good', 'Fair', 'Poor'                                   |
| phys_health_not_good      | object    | modeling_dataset.csv  | Days of poor physical health in the past 30 days                                                                |
| mental_health_not_good    | object    | modeling_dataset.csv  | Days of poor mental health in the past 30 days                                                                  |
| last_routine_checkup      | object    | modeling_dataset.csv  | Time since last routine checkup: 'Within past year', 'Within past 2 years', 'Within past 5 years', '>5 years ago' |
| visit_dentist_past_year   | object    | modeling_dataset.csv  | Visit to dentist in past year                                                                                   |
| health_insurance          | object    | modeling_dataset.csv  | Presence of health insurance                                                                                    |
| phys_health_past_30_days  | object    | modeling_dataset.csv  | Participation in physical activity                                                                              |
| stroke                    | object    | modeling_dataset.csv  | History of stroke diagnosis                                                                                     |
| cancer                    | object    | modeling_dataset.csv  | History of cancer diagnosis                                                                                     |
| kidney_disease            | object    | modeling_dataset.csv  | History of kidney disease diagnosis                                                                             |
| colon_sigmoidoscopy       | object    | modeling_dataset.csv  | History of colonoscopy/sigmoidoscopy examination                                                                       |
| asthma_status             | object    | modeling_dataset.csv  | History of asthma diagnosis                                                                                     |
| race_ethnicity            | object    | modeling_dataset.csv  | Ethnicity of respondent: 'White', 'Black', 'American Indian/ Alaskan Native', 'Asian', 'Native Hawaiian/ Pacific Islander', 'Multiracial', 'Hispanic' |
| sex                       | object    | modeling_dataset.csv  | Gender of respondent                                                                         |
| education                 | object    | modeling_dataset.csv  | Education level of respondent: 'Did not grad High Sch', 'Grad High Sch', 'Attended College or Tech Sch', 'Grad College or Tech Sch' |
| smoker_status             | object    | modeling_dataset.csv  | Smoking status of respondent: 'Current smoker - every day', 'Current smoker - some days', 'Former smoker', 'Never smoked' |
| e_cig_smoker             | object    | modeling_dataset.csv  | E-cigarette usage of respondent                                                                   |
| binge_drinker             | object    | modeling_dataset.csv  | Binge drinking behavior                                                                                         |
| heavy_drinker             | object    | modeling_dataset.csv  | Heavy drinking behavior                                                                                         |
| income_groups             | object    | modeling_dataset.csv  | Annual income (USD) split into groups: '<50K', '50K-100K', '100K-150K', '150K - 200K', '>200K'               |

## Python Libraries Requirements

For the purpose of this project, installation of the following libraries is required:

1. `imblearn`
2. `keras`
3. `lightgbm`
4. `math`
5. `matplotlib.pyplot`
6. `missforest`
7. `missingno`
8. `numpy`
9. `pandas`
10. `scikit-learn`
11. `seaborn`
12. `time`
13. `tensorflow`
14. `xgboost`