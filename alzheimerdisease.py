# -*- coding: utf-8 -*-
"""
# ALZHEIMER'S DISEASE CLASSIFICATION
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("alzheimers_disease_data.csv")
df

"""# Data Overview and Data Cleaning"""

print("\nDataSet Dimensions:")
print(df.shape)

print("\nDataset Info:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing values per column:\n", df.isnull().sum())

print("\nDuplicate Rows:\n",df.duplicated().sum())

"""- The dataset has 2149 rows and 35 columns.
- The schema shows a mix of numerical, categorical, and binary indicators.
- No major datatype issues are visible — data types are appropriate for EDA and modeling.
- Summary statistics indicate the dataset is well-structured and ready for preprocessing.
- Missing values and duplicate rows checks help ensure data quality early in the workflow.
"""

# Checking for binary features that dataset contains only [0,1]
binary_cols = [c for c in df.columns if df[c].nunique() == 2]
for col in binary_cols:
    print(col, df[col].unique())

df.describe().transpose().head(35)

# Drop unnecessary columns
drop_cols = ["DoctorInCharge", "PatientID"]

for col in drop_cols:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

df.head()

"""- Columns DoctorInCharge and PatientID were removed because they do not contribute to diagnosis prediction.
- These are identifiers / administrative attributes and would introduce noise or leakage.
- Clean feature groups help simplify the modeling pipeline.
"""

# Separate Feature Groups as dataset appears
Patient_Info = df[['Age', 'Gender', 'Ethnicity', 'EducationLevel', 'Diagnosis']]

Lifestyle_Factors = df[['BMI','Smoking','AlcoholConsumption','PhysicalActivity',
                        'DietQuality','SleepQuality','Diagnosis']]

Medical_History = df[['FamilyHistoryAlzheimers','CardiovascularDisease',
                      'Diabetes','Depression','HeadInjury','Hypertension','Diagnosis']]

Clinical_Measurements = df[['SystolicBP','DiastolicBP','CholesterolLDL',
                            'CholesterolHDL','CholesterolTotal',
                            'CholesterolTriglycerides','Diagnosis']]

Cognitive_Assessments = df[['MMSE','FunctionalAssessment','MemoryComplaints',
                            'BehavioralProblems','ADL','Confusion','Disorientation',
                            'PersonalityChanges','DifficultyCompletingTasks',
                            'Forgetfulness','Diagnosis']]

"""- Features are grouped into logical sets:

  - Patient Info : demographic characteristics

  - Lifestyle Factors : behavior-related risk variables

  - Medical History : chronic or previous illnesses

  - Clinical Measurements : physiological metrics

  - Cognitive Assessments : psychological + neurocognitive indicators

This helps simplify EDA and model interpretation.
"""

# Diagnosis Distribution [Target Variable]
plt.figure(figsize=(6,4))
ax = sns.countplot(data=df, x='Diagnosis', palette='Set2')

for p in ax.patches:
    count = int(p.get_height())
    ax.text(
        p.get_x() + p.get_width()/2,
        p.get_height() + 2,
        count,
        ha='center', fontsize=12, color='black'
    )

plt.title("Diagnosis Distribution (0 = No Alzheimer’s, 1 = Alzheimer’s)")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.show()

print("\nPercentage Distribution:")
print(df['Diagnosis'].value_counts(normalize=True) * 100)

"""- Diagnosis distribution shows potential class imbalance.

- If one class dominates, models may get biased; sampling techniques or class weights may be needed.

- Visualizing percentages helps understand dataset representation and fairness.

# Data Visualizations
"""

# Histogram visualization [Univariate]
num_cols = df.select_dtypes(include=['int64','float64']).columns
cols = 6
rows = int(np.ceil(len(num_cols) / cols))

plt.figure(figsize=(18, rows*3))

for i, col in enumerate(num_cols):
    plt.subplot(rows, cols, i+1)
    sns.histplot(df[col], kde=True, color='teal')
    plt.title(col)

plt.tight_layout()
plt.show()

"""- Distributions reveal skewness in many variables (e.g., BP, cholesterol, cognitive scores).
- Several variables may benefit from normalization or scaling before modeling.
- Some clinical measurements show near-normal distribution, which is ideal for ML algorithms.
"""

# Count plots for all binary variables
binary_cols = [col for col in df.columns if df[col].nunique() == 2 and col != 'Diagnosis']
plt.figure(figsize=(16, len(binary_cols) * 3))

for i, col in enumerate(binary_cols, 1):
    plt.subplot(len(binary_cols), 2, i)
    ax = sns.countplot(data=df, x=col, hue='Diagnosis', palette='Set2')

    for container in ax.containers:
        ax.bar_label(container, fontsize=9)

    plt.title(f"{col} vs Diagnosis")
    plt.xlabel(f"{col} (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    plt.legend(title="Diagnosis", labels=["0 = No AD", "1 = AD"])

plt.tight_layout()
plt.show()

"""- Many binary features (FamilyHistory, Smokes, Diabetes, etc.) show visible differences between Diagnosis = 0 and 1.

- A higher count of positive risk factors appears in Alzheimer’s patients.

- These variables are likely important predictors for classification.


"""

# Scatter plots visualization for all numerical variables[Bivariate visualizations]
num_cols = df.select_dtypes(include=['int64','float64']).columns

n_cols = len(num_cols)

cols = 6
rows = int(np.ceil(n_cols / cols))

plt.figure(figsize=(16, rows * 3))
for i, col in enumerate(num_cols):
    plt.subplot(rows, cols, i + 1)

    jitter = np.random.uniform(-0.1, 0.1, size=len(df))
    sns.scatterplot(
        x=df['Diagnosis'] + jitter,
        y=df[col],
        alpha=0.5,
        color='teal',
        s=20
    )

    plt.xlabel("Diagnosis (0 = No, 1 = Yes)")
    plt.ylabel(col)

plt.tight_layout()
plt.show()

"""- Scatter plots across all numerical variables with jittering show visible trends where Alzheimer’s (1) clusters differently.

- Cognitive scores (MMSE, ADL, etc.) show strong separation.

- Clinical measures show weaker separation, indicating lower predictive strength.
"""

corr = df.corr()
corr

# Multi variate visualization of each Feature group
sns.pairplot(
    Patient_Info,
    hue = 'Diagnosis',
    diag_kind = 'kde'
)
plt.gcf().suptitle("PairPlot: Patient Information", y=1.02, fontsize=18)
plt.show()

# Lifestyle Factors
sns.pairplot(
    Lifestyle_Factors,
    hue = 'Diagnosis',
    diag_kind = 'kde'
)
plt.gcf().suptitle("PairPlot: Lifestyle Factors", y=1.02, fontsize=18)
plt.show()

# Medical History
sns.pairplot(
    Medical_History,
    hue = 'Diagnosis',
    diag_kind = 'kde'
)
plt.gcf().suptitle("PairPlot: Medical History", y=1.02, fontsize=18)
plt.show()

# Clinical Measurements
sns.pairplot(
    Clinical_Measurements,
    hue='Diagnosis',
    diag_kind='kde'
)

plt.gcf().suptitle("PairPlot: Clinical Measurements", y=1.02, fontsize=18)
plt.show()

# Cognitive Assessments
sns.pairplot(
    Cognitive_Assessments,
    hue = 'Diagnosis',
    diag_kind = 'kde'
)
plt.gcf().suptitle("PairPlot: Cognitive and Functional Assessments", y=1.02, fontsize=20)
plt.show()

"""Patient Info

  - Age shows mild upward trend with Alzheimer’s presence.

  - Gender & Ethnicity show mixed patterns — weaker predictors.

Lifestyle Factors

  - DietQuality, PhysicalActivity, SleepQuality show separation between AD vs non-AD.

  - Poor lifestyle scores correlate with higher Alzheimer’s probability.

Medical History

  - Conditions like Depression, Diabetes, Hypertension show clear color separation → meaningful predictors.

Clinical Measurements

  - Cholesterol-based measures show modest separation.

Cognitive Assessments

  - Strong separation across all cognitive scores → highest predictive power.
"""

# Boxplots for outlier inspection
cols = 4
rows = int(np.ceil(len(num_cols) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 1.5))
axes = axes.flatten()

colors = sns.color_palette("Set2", len(num_cols))

for i, col in enumerate(num_cols):
    sns.boxplot(x=df[col], ax=axes[i], color=colors[i])
    axes[i].set_title(col)

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

"""- Some variables contain outliers (cholesterol, BP).

- Outliers may need treatment depending on modeling method.

- Tree models handle them well; normalization-based models (SVM, Logistic Regression) may require preprocessing.
"""

# violin plots
n_cols = 3
n_rows = int(np.ceil(len(num_cols) / n_cols))

plt.figure(figsize=(16, 3 * n_rows))
for i, col in enumerate(num_cols):
    plt.subplot(n_rows, n_cols, i+1)
    sns.violinplot(x=df[col], color="teal")
    plt.title(f"{col} Violin Plot")
plt.tight_layout()
plt.show()

"""- Clear separation in cognitive assessment distributions.

- Lifestyle variables have broader spread and overlap more.

- Clinical variables show typical medical variability but with mild skewness.
"""

# Correlation Heatmap for each Feature group
# heatmap for patients information
plt.figure(figsize=(8,5))
sns.heatmap(Patient_Info.select_dtypes(include=['int64','float64']).corr(),
            annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap: Patient Info")
plt.show()

# Lifestyle factors
plt.figure(figsize=(8,5))
sns.heatmap(Lifestyle_Factors.select_dtypes(include=['int64','float64']).corr(),
            annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap: Lifestyle Factors")
plt.show()

# medical history
plt.figure(figsize=(8,5))
sns.heatmap(Medical_History.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap: Medical History")
plt.show()

# clinical measurements
plt.figure(figsize=(8,5))
sns.heatmap(Clinical_Measurements.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap: Clinical Measurements")
plt.show()

# Cognitive assessments
plt.figure(figsize=(10,7))
sns.heatmap(Cognitive_Assessments.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap: Cognitive and Functional Assessment")
plt.show()

# an overall correlation heatmap of all features
plt.figure(figsize=(16,12))
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()

"""Patient Info Heatmap

  - Age correlates positively with Alzheimer’s
  - gender and ethnicity weak.

Lifestyle Factors Heatmap

  - SleepQuality, PhysicalActivity, DietQuality moderately influence outcome.

Medical History Heatmap

  - Depression, Cardiovascular disease, FamilyHistory strongly correlated.

Clinical Measurements Heatmap

  - LDL, HDL, Systolic BP show patterned correlations.

Cognitive Assessment Heatmap

  - MMSE, ADL, Memory Complaints, Confusion show the strongest correlation with Alzheimer’s.

Confirms that cognitive attributes are dominant correlators with Alzheimer’s diagnosis.

Multicollinearity is expected among related cognitive features.
"""

# feature importance based on correaltion heatmap
corr = df.corr()['Diagnosis'].drop('Diagnosis').sort_values()

colors = []
for value in corr:
    if abs(value) >= 0.20:
        colors.append('red')
    elif abs(value) >= 0.02:
        colors.append('orange')
    else:
        colors.append('green')

plt.figure(figsize=(12,10))
corr.plot(kind='barh', color=colors)
plt.title("Correlation of Features with Alzheimer's Diagnosis", fontsize=16)
plt.xlabel("Correlation Strength")
plt.ylabel("Features")
plt.grid(axis='x', linestyle='--', alpha=0.4)

import matplotlib.patches as mpatches
plt.legend(handles=[
    mpatches.Patch(color='red',   label='Highly Correlated (|corr| ≥ 0.20)'),
    mpatches.Patch(color='orange',label='Moderately Correlated (0.025 ≤ |corr| < 0.20)'),
    mpatches.Patch(color='green', label='Weakly Correlated (|corr| < 0.025)')
])

plt.show()

"""This section highlights how strongly each feature correlates with Alzheimer’s diagnosis using your custom color-coded scale:
- RED — Highly Important Features (Strong Correlation)

  - These features show the strongest statistical relationship with Alzheimer’s disease and have the highest predictive power.
- YELLOW — Moderately Important Features (Medium Correlation)

  - These features have noticeable but moderate influence. They often contribute when combined with other risk factors.
- GREEN — Low-Impact or Weakly Correlated Features

  - These features show minimal linear correlation with the diagnosis, but may still help in non-linear models
"""

# Label encode Gender & Ethnicity (as requested: option A)
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for col in ['Gender', 'Ethnicity']:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"Label encoded {col}: classes={le.classes_}")

"""# Feature Engineering, Data Partitioning"""



# Feature Engineering
selected_features = [
    'Age','Gender','Ethnicity','BMI','Smoking','AlcoholConsumption',
    'MMSE','ADL','FunctionalAssessment','MemoryComplaints',
    'Confusion','Disorientation','DifficultyCompletingTasks',
    'Forgetfulness','BehavioralProblems','PersonalityChanges',
    'SleepQuality','FamilyHistoryAlzheimers','Hypertension',
    'CardiovascularDisease','Diabetes','Depression','HeadInjury'
]


X = df[selected_features]
y = df['Diagnosis']

# Quick EDA notes
print("\nSelected features overview (first 3 rows):")
print(X.head(3))
print("\nTarget distribution:")
print(y.value_counts(), y.value_counts(normalize=True))

# Data partitioning or splitting and Scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaledX_train_scaled_df = pd.DataFrame(X_train_scaled, columns=selected_features)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=selected_features)

"""# Model Building"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

results = []
roc_curves = {}
conf_matrices = {}

def evaluate_model(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    return [name, acc, prec, rec, f1, roc_auc]

# -------------------------------
# Logistic Regression
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_proba = lr.predict_proba(X_test_scaled)[:,1]

results.append(evaluate_model("Logistic Regression", y_test, lr_pred, lr_proba))
conf_matrices["Logistic Regression"] = confusion_matrix(y_test, lr_pred)
roc_curves["Logistic Regression"] = roc_curve(y_test, lr_proba)

# -------------------------------
# KNN
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)
knn_proba = knn.predict_proba(X_test_scaled)[:,1]

results.append(evaluate_model("KNN", y_test, knn_pred, knn_proba))
conf_matrices["KNN"] = confusion_matrix(y_test, knn_pred)
roc_curves["KNN"] = roc_curve(y_test, knn_proba)

# -------------------------------
# SVM (RBF)
svm = SVC(kernel="rbf", probability=True)
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)
svm_proba = svm.predict_proba(X_test_scaled)[:,1]

results.append(evaluate_model("SVM (RBF)", y_test, svm_pred, svm_proba))
conf_matrices["SVM (RBF)"] = confusion_matrix(y_test, svm_pred)
roc_curves["SVM (RBF)"] = roc_curve(y_test, svm_proba)

# -------------------------------
# Random Forest
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
rf_proba = rf.predict_proba(X_test_scaled)[:,1]

results.append(evaluate_model("Random Forest", y_test, rf_pred, rf_proba))
conf_matrices["Random Forest"] = confusion_matrix(y_test, rf_pred)
roc_curves["Random Forest"] = roc_curve(y_test, rf_proba)

# -------------------------------
# XGBoost
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb.fit(X_train_scaled, y_train)

xgb_pred = xgb.predict(X_test_scaled)
xgb_proba = xgb.predict_proba(X_test_scaled)[:,1]

results.append(evaluate_model("XGBoost", y_test, xgb_pred, xgb_proba))
conf_matrices["XGBoost"] = confusion_matrix(y_test, xgb_pred)
roc_curves["XGBoost"] = roc_curve(y_test, xgb_proba)

# -------------------------------
# LightGBM
lgbm = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)
lgbm.fit(X_train_scaled, y_train)
lgbm_pred = lgbm.predict(X_test_scaled)
lgbm_proba = lgbm.predict_proba(X_test_scaled)[:,1]

results.append(evaluate_model("LightGBM", y_test, lgbm_pred, lgbm_proba))
conf_matrices["LightGBM"] = confusion_matrix(y_test, lgbm_pred)
roc_curves["LightGBM"] = roc_curve(y_test, lgbm_proba)

# -------------------------------
# SUMMARY RESULTS TABLE
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
).sort_values(by="ROC-AUC", ascending=False)

print("\n MODEL COMPARISON")
print(results_df)

# Confusion matrix
import seaborn as sns
plt.figure(figsize=(8, 12))

i = 1
for model_name, cm in conf_matrices.items():
    plt.subplot(3, 2, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(model_name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    i += 1

plt.tight_layout()
plt.show()

# Combined roc plot
plt.figure(figsize=(8,6))

for name, (fpr, tpr, thresholds) in roc_curves.items():
    # find the correct probability array
    proba = {
        "Logistic Regression": lr_proba,
        "KNN": knn_proba,
        "SVM (RBF)": svm_proba,
        "Random Forest": rf_proba,
        "XGBoost": xgb_proba,
        "LightGBM": lgbm_proba
    }[name]

    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0,1],[0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()
plt.show()

# Feture importnace according to the best model
fi_xgb = pd.Series(xgb.feature_importances_, index=selected_features).sort_values()
fi_xgb.plot(kind="barh", figsize=(8,6), title="XGB Feature Importance")
plt.show()

# ============================================
# SAVE TRAINED MODEL, SCALER & FEATURE LIST
# ============================================

import joblib
import json

print("\nSaving model files...")

# Save trained XGBoost model
joblib.dump(xgb, "best_model.pkl")

# Save the fitted StandardScaler
joblib.dump(scaler, "scaler.pkl")

# Save selected feature names (list)
with open("selected_features.json", "w") as f:
    json.dump(selected_features, f)

print("\n-----------------------------------")
print("✔ Saved best_model.pkl")
print("✔ Saved scaler.pkl")
print("✔ Saved selected_features.json")
print("-----------------------------------")
print("Model files saved successfully!")


def predict_alzheimer(patient_dict):
    """
    patient_dict: dictionary of input features
    Returns: predicted class (0/1), probability of Alzheimer's
    """

    # Convert dict → DataFrame
    df_input = pd.DataFrame([patient_dict])

    # Ensure correct column order
    df_input = df_input[selected_features]

    # Scale input
    df_scaled = scaler.transform(df_input)

    # Predict
    pred = xgb.predict(df_scaled)[0]
    proba = xgb.predict_proba(df_scaled)[0][1]

    return int(pred), float(proba)

case = {
    'Age': 60,
    'Gender': 0,
    'Ethnicity': 2,
    'BMI': 23.4,
    'Smoking': 1,
    'AlcoholConsumption': 1.5,
    'MMSE': 15,
    'ADL': 15,
    'FunctionalAssessment': 20,
    'MemoryComplaints': 1,
    'Confusion': 1,
    'Disorientation': 1,
    'DifficultyCompletingTasks': 1,
    'Forgetfulness': 1,
    'BehavioralProblems': 1,
    'PersonalityChanges': 1,
    'SleepQuality': 0,
    'FamilyHistoryAlzheimers': 1,
    'Hypertension': 0,
    'CardiovascularDisease': 0,
    'Diabetes': 1,
    'Depression': 0,
    'HeadInjury': 1
}

pred, proba = predict_alzheimer(case)

print("\n--- Prediction Result ---")
if pred == 1:
    print(f"🧠 Diagnosis: **Alzheimer’s Detected**")
    print(f"🔴 Probability: {proba:.2%}")
else:
    print(f"😊 Diagnosis: **No Alzheimer’s**")
    print(f"🟢 Probability of being healthy: {(1-proba):.2%}")

print("\nClinical Interpretation:")
if proba > 0.75:
    print("Very High Risk – Immediate medical follow-up recommended.")
elif proba > 0.50:
    print("Moderate Risk – Monitoring and cognitive screening advised.")
else:
    print("Low Risk – No immediate cognitive concern.")
