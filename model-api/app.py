from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:8000"])

# Load the trained model and label encoders
try:
    with open('loan_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('label_encoders.pkl', 'rb') as file:
        label_encoders = pickle.load(file)
except FileNotFoundError:
    print("Model files not found. Please run train_model.py first.")
    model = None
    label_encoders = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or label_encoders is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        
        # Create DataFrame with the received data
        input_data = pd.DataFrame([{
            'Gender': data['gender'],
            'Married': data['married'],
            'Dependents': data['dependents'],
            'Education': data['education'],
            'Self_Employed': data['selfEmployed'],
            'ApplicantIncome': data['applicantIncome'],
            'CoapplicantIncome': data['coapplicantIncome'],
            'LoanAmount': data['loanAmount'],
            'Loan_Amount_Term': data['loanAmountTerm'],
            'Credit_History': data['creditHistory'],
            'Property_Area': data['propertyArea']
        }])
        
        print(f"Input DataFrame before encoding: {input_data}")
        
        # Encode categorical variables
        for column in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
            input_data[column] = label_encoders[column].transform(input_data[column].astype(str))
        
        # Add engineered features (exactly as in training)
        input_data['Total_Income'] = input_data['ApplicantIncome'] + input_data['CoapplicantIncome']
        input_data['Income_to_Loan_Ratio'] = input_data['Total_Income'] / (input_data['LoanAmount'] + 1e-6)
        input_data['Has_Coapplicant'] = (input_data['CoapplicantIncome'] > 0).astype(int)
        input_data['High_Income'] = (input_data['Total_Income'] > 15000).astype(int)
        input_data['Low_Income'] = (input_data['Total_Income'] < 5000).astype(int)
        input_data['Very_Low_Income'] = (input_data['Total_Income'] < 3000).astype(int)
        input_data['High_Loan_Ratio'] = (input_data['Income_to_Loan_Ratio'] > 10).astype(int)
        input_data['Low_Loan_Ratio'] = (input_data['Income_to_Loan_Ratio'] < 3).astype(int)
        
        # Select features in the same order as training
        features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                   'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                   'Credit_History', 'Property_Area', 'Total_Income', 'Income_to_Loan_Ratio',
                   'Has_Coapplicant', 'High_Income', 'Low_Income', 'Very_Low_Income', 
                   'High_Loan_Ratio', 'Low_Loan_Ratio']
        
        input_data = input_data[features]
        
        print(f"Input DataFrame after encoding and feature engineering: {input_data}")
        print(f"Input data types: {input_data.dtypes}")
        print(f"Number of features: {len(input_data.columns)}")
        
        # Make prediction (Random Forest doesn't need scaling)
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        print(f"Model prediction: {prediction}")
        print(f"Model prediction probabilities: {prediction_proba}")
        
        # Convert prediction to boolean (assuming 0=N, 1=Y)
        result = bool(prediction == 1)
        print(f"Final result: {result}")
        
        return jsonify({'result': result})
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True) 