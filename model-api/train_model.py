import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# Load the data
data = pd.read_csv('loan_data.csv')

# Create a synthetic target variable based on loan eligibility criteria
# This is a simplified approach - in real scenarios, you'd have actual loan status data
def create_loan_status(row):
    # Simple eligibility criteria based on income and loan amount
    total_income = row['ApplicantIncome'] + row['CoapplicantIncome']
    loan_amount = row['LoanAmount'] if pd.notna(row['LoanAmount']) else 0
    
    # Basic eligibility rules
    if pd.isna(loan_amount) or loan_amount == 0:
        return 'N'
    
    # Income to loan ratio should be reasonable
    if total_income > 0:
        income_to_loan_ratio = total_income / loan_amount
        
        # More comprehensive eligibility criteria
        # 1. High income to loan ratio (good)
        # 2. Good credit history
        # 3. Reasonable loan term
        # 4. Graduate education (bonus)
        
        credit_good = row['Credit_History'] == 1
        education_good = row['Education'] == 'Graduate'
        loan_term_reasonable = row['Loan_Amount_Term'] <= 360  # 30 years max
        
        # Approval criteria
        if income_to_loan_ratio >= 3 and credit_good:
            return 'Y'  # Good income ratio and credit
        elif income_to_loan_ratio >= 5:
            return 'Y'  # Very good income ratio
        elif income_to_loan_ratio >= 2 and credit_good and education_good:
            return 'Y'  # Moderate ratio but good credit and education
        else:
            return 'N'
    else:
        return 'N'

# Create the target variable
data['Loan_Status'] = data.apply(create_loan_status, axis=1)

# Handle missing values in categorical columns
data['Gender'] = data['Gender'].fillna('Male')
data['Married'] = data['Married'].fillna('No')
data['Dependents'] = data['Dependents'].fillna('0')
data['Education'] = data['Education'].fillna('Graduate')
data['Self_Employed'] = data['Self_Employed'].fillna('No')
data['Property_Area'] = data['Property_Area'].fillna('Urban')

# Handle missing values in numerical columns
data['ApplicantIncome'] = data['ApplicantIncome'].fillna(data['ApplicantIncome'].median())
data['CoapplicantIncome'] = data['CoapplicantIncome'].fillna(0)
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(360)
data['Credit_History'] = data['Credit_History'].fillna(1)

# Create separate label encoders for each categorical column
label_encoders = {}
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Select features
X = data[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
          'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
          'Loan_Amount_Term', 'Credit_History', 'Property_Area']]
Y = data['Loan_Status']

# Verify no NaN values remain
print("Checking for NaN values in features:")
print(X.isnull().sum())
print("Checking for NaN values in target:")
print(Y.isnull().sum())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Save the model
with open('loan_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the label encoders
with open('label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)

print("Model trained and saved successfully!")
print(f"Training accuracy: {model.score(X_train, y_train):.3f}")
print(f"Test accuracy: {model.score(X_test, y_test):.3f}")
print(f"Target distribution: {Y.value_counts().to_dict()}")
print("Label encoders created for:", list(label_encoders.keys()))

# Test the logic with sample data
print("\nTesting loan eligibility logic:")
test_cases = [
    {'ApplicantIncome': 12341234, 'CoapplicantIncome': 0, 'LoanAmount': 12345, 'Credit_History': 1, 'Education': 'Graduate', 'Loan_Amount_Term': 213},
    {'ApplicantIncome': 5000, 'CoapplicantIncome': 2000, 'LoanAmount': 150, 'Credit_History': 1, 'Education': 'Graduate', 'Loan_Amount_Term': 360},
    {'ApplicantIncome': 3000, 'CoapplicantIncome': 0, 'LoanAmount': 100, 'Credit_History': 0, 'Education': 'Not Graduate', 'Loan_Amount_Term': 240}
]

for i, test_case in enumerate(test_cases):
    result = create_loan_status(pd.Series(test_case))
    total_income = test_case['ApplicantIncome'] + test_case['CoapplicantIncome']
    ratio = total_income / test_case['LoanAmount']
    print(f"Test {i+1}: Income={total_income}, Loan={test_case['LoanAmount']}, Ratio={ratio:.1f}, Credit={test_case['Credit_History']}, Education={test_case['Education']} -> {result}")