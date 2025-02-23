# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING
COMPANY: CODTECH IT SOLUTIONS

NAME:PRIYADHARSHINI V

INTERN ID: CT08SEW

DOMAIN:DATA ANALYTICS

DURATION: 4 WEEEKS

MENTOR: NEELA SANTOSH

DESCRIPTION:
1. Load Dataset
Reads a CSV file (tsak2intern (1).csv) containing cryptocurrency price data.
Displays the first few rows to understand the dataset.

2. Feature Engineering
Creates a new feature Price_Change (difference between consecutive closing prices).
Defines classification target (Target):
1 if price increased
0 if price decreased
Selects relevant features: Open, High, Low, Volume, Close.

3. Data Splitting & Preprocessing
Splits data into training (80%) and testing (20%) sets.
Standardizes the dataset using StandardScaler to improve model performance.

4. Regression Model (Random Forest Regressor)
Trains a Random Forest Regressor to predict future cryptocurrency prices.
Evaluates performance using Mean Absolute Error (MAE).

5. Classification Model (Random Forest Classifier)
Trains a Random Forest Classifier to predict whether prices will go up or down.
Evaluates performance using Accuracy Score.

6. Visualization
Regression Plot: Compares actual vs. predicted prices.
Classification Heatmap: Displays confusion matrix for classification results.

OUTPUT OF THE PICTURE:
