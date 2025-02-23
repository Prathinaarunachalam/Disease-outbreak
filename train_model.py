import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Sample Dataset (Replace with a real dataset)
data = {
    'temperature': [98.6, 101.3, 99.1, 102.0, 97.5, 100.2, 99.5, 101.8, 98.2, 102.5],
    'cough': [1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    'fatigue': [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    'chest_pain': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'shortness_of_breath': [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    'blood_sugar': [120, 180, 110, 200, 90, 160, 140, 190, 95, 210],
    'frequent_urination': [1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    'disease': ['Flu', 'COVID-19', 'Healthy', 'Pneumonia', 'Healthy', 'Diabetes', 'Flu', 'COVID-19', 'Healthy', 'Pneumonia'],
    'severity': ['Mild', 'Severe', 'None', 'Moderate', 'None', 'Severe', 'Mild', 'Severe', 'None', 'Moderate']
}

df = pd.DataFrame(data)

# Feature selection
X = df.drop(columns=['disease', 'severity'])
X['blood_sugar'] = (X['blood_sugar'] > 140).astype(int)  # Convert blood sugar to binary
y_disease = LabelEncoder().fit_transform(df['disease'])
y_severity = LabelEncoder().fit_transform(df['severity'])

# Split data
X_train, X_test, y_train_disease, y_test_disease = train_test_split(X, y_disease, test_size=0.2, random_state=42)
y_train_severity, y_test_severity = train_test_split(y_severity, test_size=0.2, random_state=42)

# Train models
model_disease = RandomForestClassifier(n_estimators=100, random_state=42)
model_disease.fit(X_train, y_train_disease)

model_severity = RandomForestClassifier(n_estimators=100, random_state=42)
model_severity.fit(X_train, y_train_severity)

# Save models
pickle.dump(model_disease, open('models/disease_model.pkl', 'wb'))
pickle.dump(model_severity, open('models/severity_model.pkl', 'wb'))

# Save label encoders
encoder_disease = LabelEncoder()
encoder_disease.fit(df['disease'])
pickle.dump(encoder_disease, open('models/label_encoder_disease.pkl', 'wb'))

encoder_severity = LabelEncoder()
encoder_severity.fit(df['severity'])
pickle.dump(encoder_severity, open('models/label_encoder_severity.pkl', 'wb'))

print("Models and encoders saved successfully!")
