import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Sample HR data (replace with real dataset or API data)
data = {
    'age': np.random.randint(18, 65, 1000),
    'years_at_company': np.random.randint(0, 20, 1000),
    'job_satisfaction': np.random.randint(1, 5, 1000),  # 1 (low) to 4 (high)
    'monthly_salary': np.random.uniform(3000, 15000, 1000),
    'promotion_last_5years': np.random.choice([0, 1], 1000, p=[0.7, 0.3]),
    'attrition': np.random.choice([0, 1], 1000, p=[0.85, 0.15])  # 0: Stay, 1: Leave
}
df = pd.DataFrame(data)

# Prepare features (X) and target (y)
X = df[['age', 'years_at_company', 'job_satisfaction', 'monthly_salary', 'promotion_last_5years']]
y = df['attrition']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of attrition

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Visualize attrition probability distribution
plt.figure(figsize=(10, 6))
sns.histplot(y_pred_proba, bins=20, kde=True, color='purple')
plt.title('Distribution of Attrition Probability')
plt.xlabel('Attrition Probability')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('attrition_probability.png')
plt.show()

# Feature importance visualization
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importance.values, y=feature_importance.index, palette='viridis')
plt.title('Feature Importance in Employee Attrition Prediction')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# Bonus: Example code to load data from a CSV or mock API (uncomment to use)
"""
import requests
def fetch_hr_data(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        df_api = pd.DataFrame(data)
        return df_api
    return None

# Example usage
# api_url = 'https://example.com/hr-data-api'
# hr_data = fetch_hr_data(api_url)
# if hr_data is not None:
#     X_new = hr_data[['age', 'years_at_company', 'job_satisfaction', 'monthly_salary', 'promotion_last_5years']]
#     predicted_attrition = model.predict(X_new)
#     print(f'Predicted attrition for new employees: {predicted_attrition}')
"""
