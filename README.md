# Fraud Detection in Financial Payment Systems

## Project Overview
This project focuses on detecting fraudulent transactions in a financial payment system using machine learning models. The dataset is analyzed, preprocessed, and evaluated using K-Nearest Neighbors (KNN), Random Forest, XGBoost, and an Ensemble Voting Classifier.

The objective is to build a robust fraud detection system that balances precision, recall, and overall accuracy to minimize false positives and detect fraud effectively.

## Table of Contents

Project Overview
Dataset Information
Requirements
Installation and Setup
Data Preprocessing
Model Training
Evaluation
Results
Usage
Contributors
## Dataset Information
The dataset consists of transaction details, including attributes such as:

customer: Unique customer identifier
merchant: Unique merchant identifier
amount: Transaction amount
fraud: Binary target variable indicating fraudulent transactions (1 = Fraud, 0 = Non-Fraud)
The data is processed to remove unnecessary columns and encode categorical variables for machine learning purposes.

## Requirements
The following libraries and dependencies are required to run this project:

- Python >= 3.8
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
You can install the required dependencies using pip:

bash

pip install pandas numpy matplotlib seaborn scikit-learn xgboost
## Installation and Setup

Clone the repository to your local system:
bash


git clone (https://github.com/Narendra022/Fraud-Detection-)
cd fraud-detection
Install the dependencies using the requirements above.
Ensure the dataset is stored in the following structure:
markdown
Copy code
- fraud-detection/
    - Data/
        - bs140513_032310.csv
Run the Python script to execute the project:
bash

python fraud_detection.py
## Data Preprocessing

Dropped redundant columns (zipcodeOri, zipMerchant) with only one unique value.
Encoded categorical variables (customer, merchant, category) into numerical representations.
Split the data into training and testing sets using stratified sampling.
## Model Training
The following models were implemented:

- K-Nearest Neighbors (KNN)
- Random Forest Classifier
- XGBoost Classifier
- Ensemble Voting Classifier combining all models with soft voting.

## Evaluation
The models were evaluated using the following metrics:

Confusion Matrix
Classification Report
ROC-AUC Curve
Base Benchmark: The baseline accuracy is set by predicting all transactions as non-fraudulent.

## Results

- KNN: Balanced performance with precision and recall trade-offs.
- Random Forest: High recall but relatively lower precision for fraudulent class.
- XGBoost: Achieved high precision and recall with optimized hyperparameters.
- Ensemble Classifier: Combined the strengths of the models to improve overall performance.
## Usage

Run the project script:
bash
Copy code
python fraud_detection.py
Analyze the outputs, including confusion matrices, classification reports, and ROC-AUC plots.
## Contributors

Your Name: Data preprocessing, model implementation, and evaluation.
