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
        if income_to_loan_ratio > 5 and row['Credit_History'] == 1:
            return 'Y'
        elif income_to_loan_ratio > 10:
            return 'Y'
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
model = LogisticRegression(random_state=42)
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