from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:8000"])

# Load the trained model, label encoders, scaler, and feature selector
try:
    with open('loan_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('label_encoders.pkl', 'rb') as file:
        label_encoders = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('feature_selector.pkl', 'rb') as file:
        feature_selector = pickle.load(file)
except FileNotFoundError:
    print("Model files not found. Please run train_model.py first.")
    model = None
    label_encoders = None
    scaler = None
    feature_selector = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or label_encoders is None or scaler is None or feature_selector is None:
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
        
        # Add all engineered features (exactly as in training)
        input_data['Total_Income'] = input_data['ApplicantIncome'] + input_data['CoapplicantIncome']
        input_data['Income_to_Loan_Ratio'] = input_data['Total_Income'] / (input_data['LoanAmount'] + 1e-6)
        
        # Advanced income features
        input_data['ApplicantIncome_Log'] = np.log1p(input_data['ApplicantIncome'])
        input_data['CoapplicantIncome_Log'] = np.log1p(input_data['CoapplicantIncome'])
        input_data['Total_Income_Log'] = np.log1p(input_data['Total_Income'])
        input_data['LoanAmount_Log'] = np.log1p(input_data['LoanAmount'])
        
        # Square transformations
        input_data['ApplicantIncome_Squared'] = input_data['ApplicantIncome'] ** 2
        input_data['Total_Income_Squared'] = input_data['Total_Income'] ** 2
        input_data['LoanAmount_Squared'] = input_data['LoanAmount'] ** 2
        input_data['Income_to_Loan_Ratio_Squared'] = input_data['Income_to_Loan_Ratio'] ** 2
        
        # Income categories with fine granular bins
        input_data['Income_Category_Fine'] = pd.cut(input_data['Total_Income'], 
                                                   bins=[0, 3000, 5000, 8000, 12000, 18000, 25000, np.inf], 
                                                   labels=[0, 1, 2, 3, 4, 5, 6], include_lowest=True).astype(int)
        
        # Loan amount categories with fine granular bins
        input_data['LoanAmount_Category_Fine'] = pd.cut(input_data['LoanAmount'], 
                                                       bins=[0, 50, 100, 150, 250, 350, 500, np.inf], 
                                                       labels=[0, 1, 2, 3, 4, 5, 6], include_lowest=True).astype(int)
        
        # Ratio-based features
        input_data['Income_to_Loan_Ratio_Log'] = np.log1p(input_data['Income_to_Loan_Ratio'])
        input_data['High_Income_to_Loan_Ratio'] = (input_data['Income_to_Loan_Ratio'] > 10).astype(int)
        input_data['Low_Income_to_Loan_Ratio'] = (input_data['Income_to_Loan_Ratio'] < 3).astype(int)
        input_data['Optimal_Income_to_Loan_Ratio'] = ((input_data['Income_to_Loan_Ratio'] >= 3) & (input_data['Income_to_Loan_Ratio'] <= 10)).astype(int)
        input_data['Very_High_Income_to_Loan_Ratio'] = (input_data['Income_to_Loan_Ratio'] > 20).astype(int)
        input_data['Very_Low_Income_to_Loan_Ratio'] = (input_data['Income_to_Loan_Ratio'] < 1.5).astype(int)
        
        # Binary features
        input_data['Has_Coapplicant'] = (input_data['CoapplicantIncome'] > 0).astype(int)
        input_data['High_Income'] = (input_data['Total_Income'] > 15000).astype(int)
        input_data['Low_Income'] = (input_data['Total_Income'] < 5000).astype(int)
        input_data['Very_Low_Income'] = (input_data['Total_Income'] < 3000).astype(int)
        input_data['Extremely_Low_Income'] = (input_data['Total_Income'] < 2000).astype(int)
        input_data['High_Loan_Amount'] = (input_data['LoanAmount'] > 300).astype(int)
        input_data['Low_Loan_Amount'] = (input_data['LoanAmount'] < 100).astype(int)
        input_data['Very_High_Loan_Amount'] = (input_data['LoanAmount'] > 500).astype(int)
        
        # Advanced interaction features
        input_data['Credit_Income_Interaction'] = input_data['Credit_History'] * input_data['Total_Income_Log']
        input_data['Education_Income_Interaction'] = input_data['Education'] * input_data['Total_Income_Log']
        input_data['Married_Income_Interaction'] = input_data['Married'] * input_data['Total_Income_Log']
        input_data['Credit_Loan_Interaction'] = input_data['Credit_History'] * input_data['LoanAmount_Log']
        input_data['Income_Loan_Interaction'] = input_data['Total_Income_Log'] * input_data['LoanAmount_Log']
        
        # Term-based features
        input_data['Long_Term_Loan'] = (input_data['Loan_Amount_Term'] > 360).astype(int)
        input_data['Short_Term_Loan'] = (input_data['Loan_Amount_Term'] < 180).astype(int)
        input_data['Standard_Term_Loan'] = ((input_data['Loan_Amount_Term'] >= 180) & (input_data['Loan_Amount_Term'] <= 360)).astype(int)
        input_data['Very_Long_Term_Loan'] = (input_data['Loan_Amount_Term'] > 480).astype(int)
        input_data['Very_Short_Term_Loan'] = (input_data['Loan_Amount_Term'] < 120).astype(int)
        
        # Property area encoding
        input_data['Urban_Area'] = (input_data['Property_Area'] == 2).astype(int)  # Assuming Urban is encoded as 2
        input_data['Semiurban_Area'] = (input_data['Property_Area'] == 1).astype(int)  # Assuming Semiurban is encoded as 1
        input_data['Rural_Area'] = (input_data['Property_Area'] == 0).astype(int)  # Assuming Rural is encoded as 0
        
        # Dependents encoding
        input_data['No_Dependents'] = (input_data['Dependents'] == 0).astype(int)
        input_data['Has_Dependents'] = (input_data['Dependents'] != 0).astype(int)
        input_data['Many_Dependents'] = (input_data['Dependents'] == 3).astype(int)  # Assuming 3+ is encoded as 3
        input_data['Single_Dependent'] = (input_data['Dependents'] == 1).astype(int)
        input_data['Two_Dependents'] = (input_data['Dependents'] == 2).astype(int)
        
        # Risk-based features
        input_data['High_Risk_Profile'] = ((input_data['Credit_History'] == 0) & (input_data['Total_Income'] < 5000)).astype(int)
        input_data['Low_Risk_Profile'] = ((input_data['Credit_History'] == 1) & (input_data['Total_Income'] > 15000)).astype(int)
        input_data['Moderate_Risk_Profile'] = ((input_data['Credit_History'] == 1) & (input_data['Total_Income'] >= 5000) & (input_data['Total_Income'] <= 15000)).astype(int)
        
        # Employment-based features
        input_data['Self_Employed_High_Income'] = ((input_data['Self_Employed'] == 1) & (input_data['Total_Income'] > 10000)).astype(int)
        input_data['Salaried_Low_Income'] = ((input_data['Self_Employed'] == 0) & (input_data['Total_Income'] < 5000)).astype(int)
        
        # Gender-based features
        input_data['Male_High_Income'] = ((input_data['Gender'] == 1) & (input_data['Total_Income'] > 12000)).astype(int)
        input_data['Female_Low_Income'] = ((input_data['Gender'] == 0) & (input_data['Total_Income'] < 4000)).astype(int)
        
        # Select features in the same order as training
        features = [
            # Original features
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
            'Credit_History', 'Property_Area',
            
            # Engineered features
            'Total_Income', 'Income_to_Loan_Ratio', 'ApplicantIncome_Log', 'CoapplicantIncome_Log',
            'Total_Income_Log', 'LoanAmount_Log', 'Income_Category_Fine', 'LoanAmount_Category_Fine',
            'Income_to_Loan_Ratio_Log', 'High_Income_to_Loan_Ratio', 'Low_Income_to_Loan_Ratio',
            'Optimal_Income_to_Loan_Ratio', 'Very_High_Income_to_Loan_Ratio', 'Very_Low_Income_to_Loan_Ratio',
            'Has_Coapplicant', 'High_Income', 'Low_Income', 'Very_Low_Income', 'Extremely_Low_Income',
            'High_Loan_Amount', 'Low_Loan_Amount', 'Very_High_Loan_Amount', 'Credit_Income_Interaction',
            'Education_Income_Interaction', 'Married_Income_Interaction', 'Credit_Loan_Interaction',
            'Income_Loan_Interaction', 'Long_Term_Loan', 'Short_Term_Loan', 'Standard_Term_Loan',
            'Very_Long_Term_Loan', 'Very_Short_Term_Loan', 'Urban_Area', 'Semiurban_Area', 'Rural_Area',
            'No_Dependents', 'Has_Dependents', 'Many_Dependents', 'Single_Dependent', 'Two_Dependents',
            'High_Risk_Profile', 'Low_Risk_Profile', 'Moderate_Risk_Profile', 'Self_Employed_High_Income',
            'Salaried_Low_Income', 'Male_High_Income', 'Female_Low_Income', 'ApplicantIncome_Squared',
            'Total_Income_Squared', 'LoanAmount_Squared', 'Income_to_Loan_Ratio_Squared'
        ]
        
        input_data = input_data[features]
        
        print(f"Input DataFrame after encoding and feature engineering: {input_data}")
        print(f"Input data types: {input_data.dtypes}")
        print(f"Number of features: {len(input_data.columns)}")
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Apply feature selection
        input_data_selected = feature_selector.transform(input_data_scaled)
        
        # Make prediction
        prediction = model.predict(input_data_selected)[0]
        prediction_proba = model.predict_proba(input_data_selected)[0]
        
        print(f"Model prediction: {prediction}")
        print(f"Model prediction probabilities: {prediction_proba}")
        
        # Convert prediction to boolean (assuming 0=N, 1=Y)
        result = bool(prediction == 1)
        confidence = float(prediction_proba[prediction])
        
        print(f"Final result: {result}")
        print(f"Confidence: {confidence:.3f}")
        
        return jsonify({
            'result': result,
            'confidence': confidence,
            'probabilities': {
                'rejected': float(prediction_proba[0]),
                'approved': float(prediction_proba[1])
            }
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True) 