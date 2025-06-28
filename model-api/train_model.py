import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import VotingClassifier
from sklearn.utils import resample
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=== LOAN ELIGIBILITY PREDICTION - ULTRA-OPTIMIZED LOGISTIC REGRESSION ===")

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

# 3. Advanced Data Preprocessing
print("\n=== ADVANCED DATA PREPROCESSING ===")

# Handle missing values with more sophisticated methods
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

# For categorical variables, use mode with random sampling for variety
for col in categorical_cols:
    if data[col].isnull().sum() > 0:
        mode_value = data[col].mode()[0]
        # Fill missing values with mode
        data[col] = data[col].fillna(mode_value)
    data[col] = data[col].astype(str)

# For numeric variables, use median for robustness
for col in numeric_cols:
    if data[col].isnull().sum() > 0:
        median_value = data[col].median()
        data[col] = data[col].fillna(median_value)

print("Missing values handled successfully")

# 4. Ultra-Advanced Feature Engineering
print("\n=== ULTRA-ADVANCED FEATURE ENGINEERING ===")

# Basic income features
data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
data['Income_to_Loan_Ratio'] = data['Total_Income'] / (data['LoanAmount'] + 1e-6)

# Advanced income features
data['ApplicantIncome_Log'] = np.log1p(data['ApplicantIncome'])
data['CoapplicantIncome_Log'] = np.log1p(data['CoapplicantIncome'])
data['Total_Income_Log'] = np.log1p(data['Total_Income'])
data['LoanAmount_Log'] = np.log1p(data['LoanAmount'])

# Square and cube transformations for non-linear patterns
data['ApplicantIncome_Squared'] = data['ApplicantIncome'] ** 2
data['Total_Income_Squared'] = data['Total_Income'] ** 2
data['LoanAmount_Squared'] = data['LoanAmount'] ** 2
data['Income_to_Loan_Ratio_Squared'] = data['Income_to_Loan_Ratio'] ** 2

# Income categories with more granular bins
data['Income_Category_Fine'] = pd.cut(data['Total_Income'], 
                                     bins=[0, 3000, 5000, 8000, 12000, 18000, 25000, np.inf], 
                                     labels=[0, 1, 2, 3, 4, 5, 6], include_lowest=True).astype(int)

# Loan amount categories with more granular bins
data['LoanAmount_Category_Fine'] = pd.cut(data['LoanAmount'], 
                                         bins=[0, 50, 100, 150, 250, 350, 500, np.inf], 
                                         labels=[0, 1, 2, 3, 4, 5, 6], include_lowest=True).astype(int)

# Ratio-based features
data['Income_to_Loan_Ratio_Log'] = np.log1p(data['Income_to_Loan_Ratio'])
data['High_Income_to_Loan_Ratio'] = (data['Income_to_Loan_Ratio'] > 10).astype(int)
data['Low_Income_to_Loan_Ratio'] = (data['Income_to_Loan_Ratio'] < 3).astype(int)
data['Optimal_Income_to_Loan_Ratio'] = ((data['Income_to_Loan_Ratio'] >= 3) & (data['Income_to_Loan_Ratio'] <= 10)).astype(int)
data['Very_High_Income_to_Loan_Ratio'] = (data['Income_to_Loan_Ratio'] > 20).astype(int)
data['Very_Low_Income_to_Loan_Ratio'] = (data['Income_to_Loan_Ratio'] < 1.5).astype(int)

# Binary features
data['Has_Coapplicant'] = (data['CoapplicantIncome'] > 0).astype(int)
data['High_Income'] = (data['Total_Income'] > 15000).astype(int)
data['Low_Income'] = (data['Total_Income'] < 5000).astype(int)
data['Very_Low_Income'] = (data['Total_Income'] < 3000).astype(int)
data['Extremely_Low_Income'] = (data['Total_Income'] < 2000).astype(int)
data['High_Loan_Amount'] = (data['LoanAmount'] > 300).astype(int)
data['Low_Loan_Amount'] = (data['LoanAmount'] < 100).astype(int)
data['Very_High_Loan_Amount'] = (data['LoanAmount'] > 500).astype(int)

# Advanced interaction features
data['Credit_Income_Interaction'] = data['Credit_History'] * data['Total_Income_Log']
data['Education_Income_Interaction'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0}) * data['Total_Income_Log']
data['Married_Income_Interaction'] = data['Married'].map({'Yes': 1, 'No': 0}) * data['Total_Income_Log']
data['Credit_Loan_Interaction'] = data['Credit_History'] * data['LoanAmount_Log']
data['Income_Loan_Interaction'] = data['Total_Income_Log'] * data['LoanAmount_Log']

# Term-based features
data['Long_Term_Loan'] = (data['Loan_Amount_Term'] > 360).astype(int)
data['Short_Term_Loan'] = (data['Loan_Amount_Term'] < 180).astype(int)
data['Standard_Term_Loan'] = ((data['Loan_Amount_Term'] >= 180) & (data['Loan_Amount_Term'] <= 360)).astype(int)
data['Very_Long_Term_Loan'] = (data['Loan_Amount_Term'] > 480).astype(int)
data['Very_Short_Term_Loan'] = (data['Loan_Amount_Term'] < 120).astype(int)

# Property area encoding
data['Urban_Area'] = (data['Property_Area'] == 'Urban').astype(int)
data['Semiurban_Area'] = (data['Property_Area'] == 'Semiurban').astype(int)
data['Rural_Area'] = (data['Property_Area'] == 'Rural').astype(int)

# Dependents encoding
data['No_Dependents'] = (data['Dependents'] == '0').astype(int)
data['Has_Dependents'] = (data['Dependents'] != '0').astype(int)
data['Many_Dependents'] = (data['Dependents'] == '3+').astype(int)
data['Single_Dependent'] = (data['Dependents'] == '1').astype(int)
data['Two_Dependents'] = (data['Dependents'] == '2').astype(int)

# Risk-based features
data['High_Risk_Profile'] = ((data['Credit_History'] == 0) & (data['Total_Income'] < 5000)).astype(int)
data['Low_Risk_Profile'] = ((data['Credit_History'] == 1) & (data['Total_Income'] > 15000)).astype(int)
data['Moderate_Risk_Profile'] = ((data['Credit_History'] == 1) & (data['Total_Income'] >= 5000) & (data['Total_Income'] <= 15000)).astype(int)

# Employment-based features
data['Self_Employed_High_Income'] = ((data['Self_Employed'] == 'Yes') & (data['Total_Income'] > 10000)).astype(int)
data['Salaried_Low_Income'] = ((data['Self_Employed'] == 'No') & (data['Total_Income'] < 5000)).astype(int)

# Gender-based features
data['Male_High_Income'] = ((data['Gender'] == 'Male') & (data['Total_Income'] > 12000)).astype(int)
data['Female_Low_Income'] = ((data['Gender'] == 'Female') & (data['Total_Income'] < 4000)).astype(int)

print(f"Created {len([col for col in data.columns if col not in ['Loan_ID', 'Loan_Status']])} features")

# 5. Encode categorical variables
print("\n=== ENCODING CATEGORICAL VARIABLES ===")
label_encoders = {}

# Encode original categorical variables
for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
    print(f"{col} classes: {le.classes_}")

# Encode target variable
le_target = LabelEncoder()
data['Loan_Status'] = le_target.fit_transform(data['Loan_Status'])
label_encoders['Loan_Status'] = le_target
print(f"Loan_Status classes: {le_target.classes_}")

# 6. Select features and target
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

X = data[features]
y = data['Loan_Status']

print(f"\nFeature set: {len(features)} features")
print(f"Features: {features}")

# 7. Advanced Sampling for Better Balance
print("\n=== ADVANCED SAMPLING ===")
# Separate majority and minority classes
df_majority = data[data['Loan_Status'] == 1]
df_minority = data[data['Loan_Status'] == 0]

# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                  replace=False,
                                  n_samples=len(df_minority) * 2,  # Keep 2x minority for better balance
                                  random_state=42)

# Combine minority class with downsampled majority class
data_balanced = pd.concat([df_majority_downsampled, df_minority])

# Shuffle the data
data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Original data shape: {data.shape}")
print(f"Balanced data shape: {data_balanced.shape}")
print(f"Balanced class distribution: {data_balanced['Loan_Status'].value_counts()}")

# Use balanced data for training
X_balanced = data_balanced[features]
y_balanced = data_balanced['Loan_Status']

# 8. Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# 9. Feature Scaling
print("\n=== FEATURE SCALING ===")
scaler = RobustScaler()  # More robust to outliers than StandardScaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 10. Feature Selection
print("\n=== FEATURE SELECTION ===")
# Select top features using ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=35)  # Select top 35 features
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Get selected feature names
selected_features_mask = selector.get_support()
selected_features = [features[i] for i in range(len(features)) if selected_features_mask[i]]
print(f"Selected {len(selected_features)} features out of {len(features)}")

# 11. Class Weight Calculation
print("\n=== CLASS BALANCING ===")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
print(f"Class weights: {class_weight_dict}")

# 12. Advanced Hyperparameter Tuning
print("\n=== ADVANCED HYPERPARAMETER TUNING ===")
param_grid = {
    'C': [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [3000, 5000],
    'tol': [1e-4, 1e-5]
}

# Use stratified k-fold for better validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Randomized search for faster optimization
random_search = RandomizedSearchCV(
    LogisticRegression(random_state=42, class_weight=class_weight_dict),
    param_distributions=param_grid,
    n_iter=50,  # Try 50 random combinations
    cv=cv,
    scoring='accuracy',  # Optimize for accuracy
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train_selected, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.4f}")

# 13. Train final model with best parameters
print("\n=== TRAINING FINAL MODEL ===")
best_model = random_search.best_estimator_

# Cross-validation on full training set
cv_scores = cross_val_score(best_model, X_train_selected, y_train, cv=cv, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 14. Model Evaluation
print("\n=== MODEL EVALUATION ===")

# Predictions
y_pred = best_model.predict(X_test_selected)
y_pred_proba = best_model.predict_proba(X_test_selected)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-Score: {f1:.4f}")
print(f"Test ROC-AUC: {roc_auc:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"True Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")

# 15. Feature Importance Analysis
print("\n=== FEATURE IMPORTANCE ===")
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': np.abs(best_model.coef_[0])
}).sort_values('importance', ascending=False)

print("Top 15 Most Important Features:")
print(feature_importance.head(15))

# 16. Model Validation on Key Patterns
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
    # Create test data with all engineered features
    test_data = pd.DataFrame([{
        'Gender': 1, 'Married': 1, 'Dependents': 0, 'Education': test_case['Education'],
        'Self_Employed': 0, 'ApplicantIncome': test_case['Total_Income'], 'CoapplicantIncome': 0,
        'LoanAmount': 200, 'Loan_Amount_Term': 360, 'Credit_History': test_case['Credit_History'],
        'Property_Area': 2, 'Total_Income': test_case['Total_Income'],
        'Income_to_Loan_Ratio': test_case['Income_to_Loan_Ratio'], 'ApplicantIncome_Log': np.log1p(test_case['Total_Income']),
        'CoapplicantIncome_Log': 0, 'Total_Income_Log': np.log1p(test_case['Total_Income']), 'LoanAmount_Log': np.log1p(200),
        'Income_Category_Fine': 4 if test_case['Total_Income'] > 15000 else 3, 'LoanAmount_Category_Fine': 3,
        'Income_to_Loan_Ratio_Log': np.log1p(test_case['Income_to_Loan_Ratio']), 'High_Income_to_Loan_Ratio': int(test_case['Income_to_Loan_Ratio'] > 10),
        'Low_Income_to_Loan_Ratio': int(test_case['Income_to_Loan_Ratio'] < 3), 'Optimal_Income_to_Loan_Ratio': int(3 <= test_case['Income_to_Loan_Ratio'] <= 10),
        'Very_High_Income_to_Loan_Ratio': int(test_case['Income_to_Loan_Ratio'] > 20), 'Very_Low_Income_to_Loan_Ratio': int(test_case['Income_to_Loan_Ratio'] < 1.5),
        'Has_Coapplicant': 0, 'High_Income': int(test_case['Total_Income'] > 15000), 'Low_Income': int(test_case['Total_Income'] < 5000),
        'Very_Low_Income': int(test_case['Total_Income'] < 3000), 'Extremely_Low_Income': int(test_case['Total_Income'] < 2000),
        'High_Loan_Amount': 0, 'Low_Loan_Amount': 0, 'Very_High_Loan_Amount': 0,
        'Credit_Income_Interaction': test_case['Credit_History'] * np.log1p(test_case['Total_Income']),
        'Education_Income_Interaction': test_case['Education'] * np.log1p(test_case['Total_Income']),
        'Married_Income_Interaction': 1 * np.log1p(test_case['Total_Income']), 'Credit_Loan_Interaction': test_case['Credit_History'] * np.log1p(200),
        'Income_Loan_Interaction': np.log1p(test_case['Total_Income']) * np.log1p(200), 'Long_Term_Loan': 0, 'Short_Term_Loan': 0,
        'Standard_Term_Loan': 1, 'Very_Long_Term_Loan': 0, 'Very_Short_Term_Loan': 0, 'Urban_Area': 0, 'Semiurban_Area': 0, 'Rural_Area': 1,
        'No_Dependents': 1, 'Has_Dependents': 0, 'Many_Dependents': 0, 'Single_Dependent': 0, 'Two_Dependents': 0,
        'High_Risk_Profile': int((test_case['Credit_History'] == 0) and (test_case['Total_Income'] < 5000)),
        'Low_Risk_Profile': int((test_case['Credit_History'] == 1) and (test_case['Total_Income'] > 15000)),
        'Moderate_Risk_Profile': int((test_case['Credit_History'] == 1) and (test_case['Total_Income'] >= 5000) and (test_case['Total_Income'] <= 15000)),
        'Self_Employed_High_Income': 0, 'Salaried_Low_Income': int(test_case['Total_Income'] < 5000),
        'Male_High_Income': int(test_case['Total_Income'] > 12000), 'Female_Low_Income': int(test_case['Total_Income'] < 4000),
        'ApplicantIncome_Squared': test_case['Total_Income'] ** 2, 'Total_Income_Squared': test_case['Total_Income'] ** 2,
        'LoanAmount_Squared': 200 ** 2, 'Income_to_Loan_Ratio_Squared': test_case['Income_to_Loan_Ratio'] ** 2
    }])
    
    # Scale and select features
    test_data_scaled = scaler.transform(test_data)
    test_data_selected = selector.transform(test_data_scaled)
    
    prediction = best_model.predict(test_data_selected)[0]
    probability = best_model.predict_proba(test_data_selected)[0]
    
    print(f"Test {i+1}: Credit={test_case['Credit_History']}, Income={test_case['Total_Income']}, "
          f"Ratio={test_case['Income_to_Loan_Ratio']:.1f} -> Prediction: {prediction} "
          f"(Prob: {probability[prediction]:.3f})")

# 17. Save model and components
print("\n=== SAVING MODEL ===")
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('feature_selector.pkl', 'wb') as f:
    pickle.dump(selector, f)

print("Model, encoders, scaler, and feature selector saved successfully!")
print(f"Class balance: {np.bincount(y_balanced)} (0={label_encoders['Loan_Status'].classes_[0]}, 1={label_encoders['Loan_Status'].classes_[1]})")

# 18. Summary
print("\n=== MODEL SUMMARY ===")
print(f"Algorithm: Logistic Regression")
print(f"Best C parameter: {best_model.C}")
print(f"Best penalty: {best_model.penalty}")
print(f"Best solver: {best_model.solver}")
print(f"Number of original features: {len(features)}")
print(f"Number of selected features: {len(selected_features)}")
print(f"Final Test Accuracy: {accuracy:.4f}")
print(f"Final Test F1-Score: {f1:.4f}")
print(f"Final Test ROC-AUC: {roc_auc:.4f}")

if accuracy >= 0.85:
    print("✅ Target accuracy of 85%+ achieved!")
else:
    print("⚠️ Target accuracy not reached, but model is highly optimized for the given data.")