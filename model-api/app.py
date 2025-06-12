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
        data = request.json
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame([{
            'Gender': data.get('gender', 'Male'),
            'Married': data.get('married', 'No'),
            'Dependents': data.get('dependents', '0'),
            'Education': data.get('education', 'Graduate'),
            'Self_Employed': data.get('selfEmployed', 'No'),
            'ApplicantIncome': data.get('applicantIncome', 0),
            'CoapplicantIncome': data.get('coapplicantIncome', 0),
            'LoanAmount': data.get('loanAmount', 0),
            'Loan_Amount_Term': data.get('loanAmountTerm', 360),
            'Credit_History': data.get('creditHistory', 1),
            'Property_Area': data.get('propertyArea', 'Urban')
        }])
        
        # Preprocess the data using separate encoders for each categorical column
        input_data['Gender'] = label_encoders['Gender'].transform(input_data['Gender'].fillna('Male'))
        input_data['Married'] = label_encoders['Married'].transform(input_data['Married'].fillna('No'))
        input_data['Dependents'] = label_encoders['Dependents'].transform(input_data['Dependents'].fillna('0'))
        input_data['Education'] = label_encoders['Education'].transform(input_data['Education'])
        input_data['Self_Employed'] = label_encoders['Self_Employed'].transform(input_data['Self_Employed'].fillna('No'))
        input_data['Property_Area'] = label_encoders['Property_Area'].transform(input_data['Property_Area'])
        input_data['Loan_Amount_Term'] = input_data['Loan_Amount_Term'].fillna(360)
        input_data['Credit_History'] = input_data['Credit_History'].fillna(1)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return jsonify({'result': bool(prediction == 'Y')})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True) 