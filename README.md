# Fraud Detection in Financial Payment Systems
##Project Overview
This project focuses on detecting fraudulent transactions in a financial payment system using machine learning models. The dataset is analyzed, preprocessed, and evaluated using K-Nearest Neighbors (KNN), Random Forest, XGBoost, and an Ensemble Voting Classifier.

The objective is to build a robust fraud detection system that balances precision, recall, and overall accuracy to minimize false positives and detect fraud effectively.


Table of Contents
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
License
Dataset Information
The dataset used in this project is a synthetic dataset of financial transactions with fraud labels. It contains:

Columns: customer, merchant, category, amount, gender, zipcodeOri, zipMerchant, fraud, etc.
Target Variable: fraud (1 for fraudulent transactions, 0 for non-fraudulent).
Number of Samples: Over 500,000 records.
The dataset can be downloaded here (provide link if available).

Requirements
To run this project, ensure the following libraries and packages are installed:

Python (3.7 or higher)
NumPy
Pandas
Seaborn
Matplotlib
scikit-learn
XGBoost
Jupyter Notebook (optional, for running interactively)
Install all dependencies using the following command:

bash
Copy code
pip install numpy pandas seaborn matplotlib scikit-learn xgboost
Installation and Setup
Clone the Repository
Clone this repository to your local machine:

bash
Copy code
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
Install Dependencies
Install all necessary Python libraries:

bash
Copy code
pip install -r requirements.txt
Prepare Dataset

Download the dataset and place it in the Data folder.
Verify the file path in the code:
python
Copy code
data = pd.read_csv("Data/synthetic-data-from-a-financial-payment-system/bs140513_032310.csv")
Run the Project
Use Jupyter Notebook or a Python IDE to run the script step by step.

bash
Copy code
jupyter notebook fraud_detection.ipynb
Data Preprocessing
Removed redundant features like zipcodeOri and zipMerchant.
Encoded categorical variables (customer, merchant, category, gender) into numeric values.
Split the dataset into training and testing sets using stratified sampling.
Model Training
The following machine learning models were implemented:

K-Nearest Neighbors (KNN)
Random Forest Classifier
XGBoost Classifier
Voting Ensemble Classifier
The ensemble model combines predictions from individual classifiers to achieve better performance.

Evaluation
Each model is evaluated using:

Confusion Matrix
Classification Report (Precision, Recall, F1-Score)
ROC-AUC Curve
Custom function plot_roc_auc() is used to plot the ROC curve for visual evaluation.

Results
Model	Precision	Recall	F1-Score	AUC
KNN	0.80	0.65	0.72	0.85
Random Forest	0.82	0.98	0.89	0.95
XGBoost	0.90	0.91	0.90	0.97
Ensemble Classifier	0.88	0.95	0.91	0.98
The Ensemble Voting Classifier achieved the best balance between recall and precision, minimizing false positives while detecting fraud effectively.

Usage
Clone the repository and install dependencies.
Replace the dataset with your own financial transaction dataset.
Run the fraud_detection.py script or the notebook.
Evaluate the models and adjust hyperparameters as needed.
Contributors
Your Name
LinkedIn: Your LinkedIn
GitHub: Your GitHub
Feel free to contribute by submitting issues or pull requests!

License
This project is licensed under the MIT License.

Additional Files
fraud_detection.py: Main script file.
requirements.txt: List of required Python libraries.
Data/: Folder to store your dataset.
figures/: Output plots for visualizations.
This README ensures clarity, ease of use, and reusability. Let me know if you need further refinements or additional files like a requirements.txt or starter script (fraud_detection.py)!
