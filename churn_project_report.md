# Customer Churn Prediction System

## 1. Objective
Analyze customer data, build machine learning models, and identify factors associated with customer churn.

## 2. Dataset Summary
- Total customers: 7043
- Customers who stayed: 5174
- Customers who churned: 1869
- Churn rate: 26.54%

**Dataset note:** This project currently uses the IBM Telco Customer Churn dataset, which represents telecom customers rather than e-commerce customers.

## 3. Methodology
1. Data cleaning and missing-value handling
2. Exploratory data analysis
3. Categorical feature encoding
4. Training and testing split
5. Training four classification models
6. Random Forest hyperparameter tuning
7. Feature importance analysis

## 4. Model Comparison

              Model  Accuracy  Precision  Recall  F1-Score  ROC-AUC
Logistic Regression     0.738      0.504   0.783     0.614    0.842
      Random Forest     0.776      0.569   0.642     0.603    0.827
  Gradient Boosting     0.798      0.654   0.505     0.570    0.842
      Decision Tree     0.737      0.504   0.484     0.494    0.656

## 5. Tuned Random Forest Results

- Best parameters: {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
- Cross-validation F1-score: 0.635
- Accuracy: 0.761
- Precision: 0.534
- Recall: 0.781
- F1-Score: 0.634
- ROC-AUC: 0.842

## 6. Top 15 Feature Importances

                             Feature  Importance
                              tenure      0.1731
                        TotalCharges      0.1381
                   Contract_Two year      0.1102
                      MonthlyCharges      0.1001
         InternetService_Fiber optic      0.0690
      PaymentMethod_Electronic check      0.0536
                   Contract_One year      0.0424
                  OnlineSecurity_Yes      0.0370
                     TechSupport_Yes      0.0233
     TechSupport_No internet service      0.0198
DeviceProtection_No internet service      0.0187
                PaperlessBilling_Yes      0.0184
 StreamingMovies_No internet service      0.0181
                    OnlineBackup_Yes      0.0145
                      Dependents_Yes      0.0136

## 7. Business Recommendations
- Improve onboarding and early support for new customers.
- Review monthly charges and explain billing clearly.
- Investigate churn patterns across contract types.
- Check service reliability and technical support satisfaction.
- Use churn predictions to prioritize customer retention outreach.

## 8. Conclusion
The tuned Random Forest model was evaluated on a held-out test set. Its results can help prioritize customers for retention activities. Further validation and cost-benefit analysis are recommended before real-world deployment.
