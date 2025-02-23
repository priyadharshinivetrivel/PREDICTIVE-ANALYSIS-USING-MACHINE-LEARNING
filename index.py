import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

# Load the dataset (Replace 'crypto_data.csv' with actual file)
df = pd.read_csv("/content/tsak2intern (1).csv")

print("Dataset Preview:",df.head())

# Feature Engineering: Selecting important features (Assuming 'Close' is the target)
df['Price_Change'] = df['Close'].diff()
df['Target'] = np.where(df['Price_Change'] > 0, 1, 0)  # Classification target (1 = Up, 0 = Down)
df.dropna(inplace=True)

features = ['Open', 'High', 'Low', 'Volume','Close']
X = df[features]
y_reg = df['Close']  # Target for Regression
y_cls = df['Target']  # Target for Classification


# Split data
X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X, y_reg, y_cls, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Regression Model
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_reg_train)
reg_preds = reg_model.predict(X_test)
reg_mae = mean_absolute_error(y_reg_test, reg_preds)
print(f"Regression MAE: {reg_mae}")

# Train Classification Model
cls_model = RandomForestClassifier(n_estimators=100, random_state=42)
cls_model.fit(X_train, y_cls_train)
cls_preds = cls_model.predict(X_test)
cls_acc = accuracy_score(y_cls_test, cls_preds)
print(f"Classification Accuracy: {cls_acc}")

# Plot Regression Predictions
plt.figure(figsize=(10, 5))
plt.plot(y_reg_test.values, label='Actual Price', color='blue')
plt.plot(reg_preds, label='Predicted Price', color='red')
plt.title('Cryptocurrency Price Prediction')
plt.legend()
plt.show()

# Plot Classification Results
sns.heatmap(pd.crosstab(y_cls_test, cls_preds), annot=True, fmt='d', cmap='Blues')
plt.title('Classification Confusion Matrix')
plt.show()


