from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load models and encoders with error handling
try:
    model_disease = pickle.load(open('models/disease_model.pkl', 'rb'))
    model_severity = pickle.load(open('models/severity_model.pkl', 'rb'))
    encoder_disease = pickle.load(open('models/label_encoder_disease.pkl', 'rb'))
    encoder_severity = pickle.load(open('models/label_encoder_severity.pkl', 'rb'))
except Exception as e:
    logging.error(f"Error loading models: {e}")
    model_disease, model_severity, encoder_disease, encoder_severity = None, None, None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure models are loaded
        if not all([model_disease, model_severity, encoder_disease, encoder_severity]):
            return jsonify({'error': 'Models not loaded. Check server logs for details.'}), 500
        
        # Retrieve form inputs with validation
        try:
            temperature = float(request.form['temperature'])
            cough = int(request.form['cough'])
            fatigue = int(request.form['fatigue'])
            chest_pain = int(request.form['chest_pain'])
            shortness_of_breath = int(request.form['shortness_of_breath'])
            blood_sugar = float(request.form['blood_sugar'])
            frequent_urination = int(request.form['frequent_urination'])

            # Ensure binary values are valid
            if any(x not in [0, 1] for x in [cough, fatigue, chest_pain, shortness_of_breath, frequent_urination]):
                return jsonify({'error': 'Invalid input: Binary fields must be 0 or 1.'}), 400
            
            # Process blood sugar into a binary feature
            blood_sugar_high = 1 if blood_sugar > 140 else 0
        except ValueError as ve:
            return jsonify({'error': 'Invalid input: Ensure all values are correctly formatted.'}), 400

        input_data = [temperature, cough, fatigue, chest_pain, shortness_of_breath, blood_sugar_high, frequent_urination]

        # Make predictions
        disease_prediction = model_disease.predict([input_data])[0]
        severity_prediction = model_severity.predict([input_data])[0]

        disease_name = encoder_disease.inverse_transform([disease_prediction])[0]
        severity_level = encoder_severity.inverse_transform([severity_prediction])[0]

        logging.info(f"Prediction: Disease={disease_name}, Severity={severity_level}")

        return render_template('result.html', disease=disease_name, severity=severity_level)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Internal Server Error. Check logs for details.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
