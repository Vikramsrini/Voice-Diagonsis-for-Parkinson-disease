import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv(r'C:\Users\vikra\Desktop\Parkinson\Parkinsson disease.csv')

# Drop the 'name' column
df.drop(columns='name', inplace=True)

# Split features and labels
X = df.drop('status', axis=1).values
y = df['status'].values

# Check label balance
print(f"Label 1 count: {sum(y==1)}")
print(f"Label 0 count: {sum(y==0)}")

# Feature scaling
scaler = MinMaxScaler((-1, 1))
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved successfully!")

# Evaluate the model
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("XGB Model Accuracy:", accuracy)

# Confusion matrix
conf = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
