import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# 1. Load the data from Excel
file_path = 'loan_data.xls'
data = pd.read_excel(file_path)

print(f"Loaded data shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")

# 2. Handle missing values
# Fill categorical with mode, numeric with median
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0]).astype(str)
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

# 3. Encode categorical variables
label_encoders = {}
for col in categorical_cols + ['Loan_Status']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
    print(f"{col} classes: {le.classes_}")

# 4. Feature engineering (optional, but helps)
data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
data['Income_to_Loan_Ratio'] = data['Total_Income'] / (data['LoanAmount'] + 1e-6)

# 5. Select features and target
features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
            'Credit_History', 'Property_Area', 'Total_Income', 'Income_to_Loan_Ratio']
X = data[features]
y = data['Loan_Status']

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Train Logistic Regression
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# 9. Save model, scaler, encoders
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# 10. Print results
print(f"Class balance: {np.bincount(y)} (0={label_encoders['Loan_Status'].classes_[0]}, 1={label_encoders['Loan_Status'].classes_[1]})")
print(f"Train accuracy: {model.score(X_train_scaled, y_train):.3f}")
print(f"Test accuracy: {model.score(X_test_scaled, y_test):.3f}")