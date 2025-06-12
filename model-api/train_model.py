import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# 1. Load the data from Excel
file_path = 'loan_data.xls'
data = pd.read_excel(file_path)

print(f"Loaded data shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")

# 2. Analyze the dataset patterns
print("\n=== DATASET ANALYSIS ===")
print(f"Loan Status Distribution:")
print(data['Loan_Status'].value_counts())
print(f"Approval Rate: {data['Loan_Status'].value_counts(normalize=True)['Y']:.3f}")

# Analyze key patterns
print("\nCredit History vs Approval:")
credit_analysis = data.groupby('Credit_History')['Loan_Status'].value_counts(normalize=True)
print(credit_analysis)

print("\nEducation vs Approval:")
education_analysis = data.groupby('Education')['Loan_Status'].value_counts(normalize=True)
print(education_analysis)

# Income analysis
data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
data['Income_to_Loan_Ratio'] = data['Total_Income'] / (data['LoanAmount'] + 1e-6)

print(f"\nHigh income (>15000) approval rate: {data[data['Total_Income'] > 15000]['Loan_Status'].value_counts(normalize=True).get('Y', 0):.3f}")
print(f"Low income (<5000) approval rate: {data[data['Total_Income'] < 5000]['Loan_Status'].value_counts(normalize=True).get('Y', 0):.3f}")

# 3. Handle missing values
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0]).astype(str)
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

# 4. Encode categorical variables
label_encoders = {}
for col in categorical_cols + ['Loan_Status']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
    print(f"{col} classes: {le.classes_}")

# 5. Feature engineering based on domain knowledge
data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
data['Income_to_Loan_Ratio'] = data['Total_Income'] / (data['LoanAmount'] + 1e-6)
data['Has_Coapplicant'] = (data['CoapplicantIncome'] > 0).astype(int)
data['High_Income'] = (data['Total_Income'] > 15000).astype(int)
data['Low_Income'] = (data['Total_Income'] < 5000).astype(int)
data['Very_Low_Income'] = (data['Total_Income'] < 3000).astype(int)
data['High_Loan_Ratio'] = (data['Income_to_Loan_Ratio'] > 10).astype(int)
data['Low_Loan_Ratio'] = (data['Income_to_Loan_Ratio'] < 3).astype(int)

# 6. Select features and target
features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
            'Credit_History', 'Property_Area', 'Total_Income', 'Income_to_Loan_Ratio',
            'Has_Coapplicant', 'High_Income', 'Low_Income', 'Very_Low_Income', 
            'High_Loan_Ratio', 'Low_Loan_Ratio']

X = data[features]
y = data['Loan_Status']

print(f"\nFeature set: {len(features)} features")
print(f"Features: {features}")

# 7. Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 8. Train Random Forest
print("\n=== TRAINING RANDOM FOREST ===")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# 9. Evaluate the model
print("\n=== MODEL EVALUATION ===")
train_accuracy = rf_model.score(X_train, y_train)
test_accuracy = rf_model.score(X_test, y_test)

print(f"Train accuracy: {train_accuracy:.3f}")
print(f"Test accuracy: {test_accuracy:.3f}")

# Classification report
y_pred = rf_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 10. Feature importance
print("\n=== FEATURE IMPORTANCE ===")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance.head(10))

# 11. Validate model behavior on key patterns
print("\n=== MODEL VALIDATION ===")

# Test cases based on real dataset patterns
test_cases = [
    # High credit score should generally be approved
    {'Credit_History': 1, 'Total_Income': 15000, 'Income_to_Loan_Ratio': 8, 'Education': 0},
    # Low credit score should generally be rejected
    {'Credit_History': 0, 'Total_Income': 8000, 'Income_to_Loan_Ratio': 4, 'Education': 0},
    # Very low income should be rejected
    {'Credit_History': 1, 'Total_Income': 2000, 'Income_to_Loan_Ratio': 1, 'Education': 0},
    # High income with good credit should be approved
    {'Credit_History': 1, 'Total_Income': 25000, 'Income_to_Loan_Ratio': 15, 'Education': 0},
]

for i, test_case in enumerate(test_cases):
    # Create test data
    test_data = pd.DataFrame([{
        'Gender': 1, 'Married': 1, 'Dependents': 0, 'Education': test_case['Education'],
        'Self_Employed': 0, 'ApplicantIncome': test_case['Total_Income'], 'CoapplicantIncome': 0,
        'LoanAmount': 200, 'Loan_Amount_Term': 360, 'Credit_History': test_case['Credit_History'],
        'Property_Area': 2, 'Total_Income': test_case['Total_Income'],
        'Income_to_Loan_Ratio': test_case['Income_to_Loan_Ratio'], 'Has_Coapplicant': 0,
        'High_Income': int(test_case['Total_Income'] > 15000), 'Low_Income': int(test_case['Total_Income'] < 5000),
        'Very_Low_Income': int(test_case['Total_Income'] < 3000), 'High_Loan_Ratio': int(test_case['Income_to_Loan_Ratio'] > 10),
        'Low_Loan_Ratio': int(test_case['Income_to_Loan_Ratio'] < 3)
    }])
    
    prediction = rf_model.predict(test_data)[0]
    probability = rf_model.predict_proba(test_data)[0]
    
    print(f"Test {i+1}: Credit={test_case['Credit_History']}, Income={test_case['Total_Income']}, "
          f"Ratio={test_case['Income_to_Loan_Ratio']:.1f} -> Prediction: {prediction} "
          f"(Prob: {probability[prediction]:.3f})")

# 12. Save model and components
print("\n=== SAVING MODEL ===")
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Model and encoders saved successfully!")
print(f"Class balance: {np.bincount(y)} (0={label_encoders['Loan_Status'].classes_[0]}, 1={label_encoders['Loan_Status'].classes_[1]})")