
import pandas as pd

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 1. Check dataset
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# 2. Check duplicates
print("\nDuplicate rows:", df.duplicated().sum())

# 3. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 4. Clean column names
df.columns = df.columns.str.strip()

# 5. Clean text values
for col in df.select_dtypes(include=["object", "str"]).columns:
    df[col] = df[col].str.strip()

# 6. Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

print("\nMissing values after conversion:")
print(df.isnull().sum())

# 7. Churn distribution
print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nData cleaning checks completed!")

import matplotlib.pyplot as plt
import seaborn as sns

# Handle 11 missing TotalCharges values
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Chart 1: Churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("churn_distribution.png", dpi=300)
plt.show()
plt.close()

# Chart 2: Churn by contract type
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("churn_by_contract.png", dpi=300)
plt.show()
plt.close()

# Chart 3: Monthly charges by churn
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig("monthly_charges_by_churn.png", dpi=300)
plt.show()
plt.close()

# Chart 4: Tenure by churn
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Customer Tenure by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")
plt.tight_layout()
plt.savefig("tenure_by_churn.png", dpi=300)
plt.show()
plt.close()

print("EDA charts completed!")


# ==========================================
# STEP 3: FEATURE ENGINEERING AND DATA PREPARATION
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("\n--- Feature Engineering ---")

# Remove customer ID because it does not help predict churn
df = df.drop(columns=["customerID"])

# Convert target column into numbers
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Separate input features (X) and target (y)
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Convert categorical columns into numerical columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Scale numerical features
scaler = StandardScaler()

numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

print("\nFeature engineering completed successfully!")

# ==========================================
# STEP 4: TRAIN AND COMPARE ML MODELS
# ==========================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("\n--- Model Training and Evaluation ---")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Train the model
    model.fit(X_train, y_train)

    # Predict churn labels
    y_pred = model.predict(X_test)

    # Predict churn probabilities for ROC-AUC
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Stayed", "Churned"],
        zero_division=0
    ))

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stayed", "Churned"],
        yticklabels=["Stayed", "Churned"]
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(f"confusion_matrix_{filename}.png", dpi=300)
    plt.show()
    plt.close()

# Compare all models
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1-Score", ascending=False)

print("\n--- MODEL COMPARISON ---")
print(results_df.round(3).to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)

print("\nModel training and comparison completed!")
print("Results saved to model_comparison.csv")

# ==========================================
# STEP 5A: HYPERPARAMETER TUNING
# ==========================================

from sklearn.model_selection import GridSearchCV

print("\n--- Hyperparameter Tuning ---")

rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1-Score:")
print(round(grid_search.best_score_, 3))

# Evaluate the tuned model on the test set
tuned_pred = best_model.predict(X_test)
tuned_prob = best_model.predict_proba(X_test)[:, 1]

print("\n--- Tuned Random Forest Test Results ---")
print("Accuracy:", round(accuracy_score(y_test, tuned_pred), 3))
print("Precision:", round(precision_score(y_test, tuned_pred, zero_division=0), 3))
print("Recall:", round(recall_score(y_test, tuned_pred, zero_division=0), 3))
print("F1-Score:", round(f1_score(y_test, tuned_pred, zero_division=0), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, tuned_prob), 3))

# ==========================================
# STEP 5B: FEATURE IMPORTANCE
# ==========================================

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": best_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- TOP 15 CHURN FACTORS ---")
print(importance_df.head(15).round(4).to_string(index=False))

# Plot top 15 features
top_features = importance_df.head(15).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.title("Top 15 Feature Importances for Customer Churn")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()
plt.close()

importance_df.to_csv("feature_importance.csv", index=False)

print("\nFeature importance saved to feature_importance.csv")

# ==========================================
# STEP 5C: SAVE THE TRAINED MODEL
# ==========================================

import joblib

joblib.dump(best_model, "best_churn_model.pkl")
joblib.dump(scaler, "churn_scaler.pkl")
joblib.dump(list(X_train.columns), "model_features.pkl")

print("\nTuned model saved successfully!")

# ==========================================
# STEP 6: FINAL PROJECT REPORT
# ==========================================

print("\n--- Generating Final Project Report ---")

# Save model comparison results
results_df.to_csv("model_comparison.csv", index=False)

# Save the tuned model's evaluation metrics
final_metrics = {
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ],
    "Score": [
        accuracy_score(y_test, tuned_pred),
        precision_score(y_test, tuned_pred, zero_division=0),
        recall_score(y_test, tuned_pred, zero_division=0),
        f1_score(y_test, tuned_pred, zero_division=0),
        roc_auc_score(y_test, tuned_prob)
    ]
}

metrics_df = pd.DataFrame(final_metrics)
metrics_df.to_csv("tuned_model_metrics.csv", index=False)

# Business recommendations
recommendations = [
    "Improve onboarding and early support for new customers.",
    "Review monthly charges and explain billing clearly.",
    "Investigate churn patterns across contract types.",
    "Check service reliability and technical support satisfaction.",
    "Use churn predictions to prioritize customer retention outreach."
]

# Generate Markdown report
with open("churn_project_report.md", "w", encoding="utf-8") as file:
    file.write("# Customer Churn Prediction System\n\n")

    file.write("## 1. Objective\n")
    file.write(
        "Analyze customer data, build machine learning models, "
        "and identify factors associated with customer churn.\n\n"
    )

    file.write("## 2. Dataset Summary\n")
    file.write(f"- Total customers: {len(df)}\n")
    file.write(f"- Customers who stayed: {(df['Churn'] == 0).sum()}\n")
    file.write(f"- Customers who churned: {(df['Churn'] == 1).sum()}\n")
    file.write(
        f"- Churn rate: {(df['Churn'].mean() * 100):.2f}%\n\n"
    )
    file.write(
        "**Dataset note:** This project currently uses the IBM "
        "Telco Customer Churn dataset, which represents telecom "
        "customers rather than e-commerce customers.\n\n"
    )

    file.write("## 3. Methodology\n")
    file.write(
        "1. Data cleaning and missing-value handling\n"
        "2. Exploratory data analysis\n"
        "3. Categorical feature encoding\n"
        "4. Training and testing split\n"
        "5. Training four classification models\n"
        "6. Random Forest hyperparameter tuning\n"
        "7. Feature importance analysis\n\n"
    )

    file.write("## 4. Model Comparison\n\n")
    file.write(results_df.round(3).to_string(index=False))
    file.write("\n\n")

    file.write("## 5. Tuned Random Forest Results\n\n")
    file.write(f"- Best parameters: {grid_search.best_params_}\n")
    file.write(f"- Cross-validation F1-score: {grid_search.best_score_:.3f}\n")
    for _, row in metrics_df.iterrows():
        file.write(f"- {row['Metric']}: {row['Score']:.3f}\n")
    file.write("\n")

    file.write("## 6. Top 15 Feature Importances\n\n")
    file.write(importance_df.head(15).round(4).to_string(index=False))
    file.write("\n\n")

    file.write("## 7. Business Recommendations\n")
    for recommendation in recommendations:
        file.write(f"- {recommendation}\n")

    file.write(
        "\n## 8. Conclusion\n"
        "The tuned Random Forest model was evaluated on a held-out "
        "test set. Its results can help prioritize customers for "
        "retention activities. Further validation and cost-benefit "
        "analysis are recommended before real-world deployment.\n"
    )

print("Final report saved: churn_project_report.md")
print("Metrics saved: tuned_model_metrics.csv")
print("Model comparison saved: model_comparison.csv")
print("\nPROJECT COMPLETED SUCCESSFULLY!")
