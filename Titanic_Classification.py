import pandas as pd
import matplotlib
# Force the interactive popup window backend for Arch Linux
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the preprocessed data
try:
    df = pd.read_csv('clean_churn_data.csv')
except FileNotFoundError:
    print("CRITICAL ERROR: 'clean_churn_data.csv' not found. Run preprocessing.py first!")
    exit()

# ==========================================
# RUBRIC SECTION 1: Data Exploration & Visuals
# ==========================================
print("--- Data Exploration Summary ---")
print(df.info())

# VISUALIZATION 1: Churn by Contract Type
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x='Churn', hue='Contract', palette='Set1')
plt.title('Customer Churn Volume by Contract Length')
plt.show() 

# ==========================================
# RUBRIC SECTION 2: Feature Selection (Correlation)
# ==========================================
# VISUALIZATION 2: Target Correlation Heatmap
plt.figure(figsize=(4, 6))
# Forces numeric_only to bypass string conversion crashes
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix[['Churn']].sort_values(by='Churn', ascending=False).head(10), 
            annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Top 10 Feature Correlations with Churn')
plt.show() 

# Define features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# RUBRIC SECTION 3: Model Selection & Justification
# ==========================================
print("\n--- Hyperparameter Tuning (GridSearchCV) ---")
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}

# GridSearch handles 3-fold cross validation automatically
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"Optimal Parameters Chosen: {grid_search.best_params_}")

print("\n--- Feature Importances ---")
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
print("Top 5 Drivers of Customer Churn:")
print(importances)

# ==========================================
# RUBRIC SECTION 4: Training & Evaluation
# ==========================================
print("\n--- Final Model Evaluation ---")
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Accuracy: {accuracy * 100:.2f}%\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))