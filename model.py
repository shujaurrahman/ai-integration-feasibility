import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

# Load the generated dataset
df = pd.read_csv('business_data.csv')

# Preprocess categorical and numeric features
label_encoders = {}
for column in ['business_type', 'complexity_level', 'data_availability']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

scaler = StandardScaler()
numeric_features = ['annual_revenue', 'employee_count', 'current_tech_investment', 
                    'customer_satisfaction', 'operational_efficiency', 
                    'workforce_skill_level', 'ai_integration_cost', 'potential_improvement']
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Split data into features and target
X = df.drop('ai_beneficial', axis=1)
y = df['ai_beneficial']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Gradient Boosting Classifier
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Beneficial", "Beneficial"], yticklabels=["Not Beneficial", "Beneficial"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve and AUC
roc_auc = roc_auc_score(y_test, y_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Feature importance plot
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance")
plt.show()

# SHAP values for interpretability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=X.columns)

# SHAP dependence plots for top features
for feature in importance_df['Feature'][:5]:  # Plotting top 5 features
    shap.dependence_plot(feature, shap_values, X_test, feature_names=X.columns)

# Step 6: Save Model and Necessary Files
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(label_encoders, "label_encoders.joblib")
joblib.dump(explainer, "explainer.joblib")
print("Model and necessary preprocessing files saved.")
