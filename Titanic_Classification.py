import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load publicly available dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# ==========================================
# RUBRIC SECTION 1: Data Pre-processing (5 Marks)
# ==========================================
# 1. Drop columns that are mostly null or irrelevant to the model
df = df.drop(['Cabin', 'Ticket', 'PassengerId', 'Name'], axis=1)

# 2. Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median()) # Fill missing ages with median
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0]) # Fill missing ports with mode

# 3. Encode categorical variables into numeric
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# ==========================================
# RUBRIC SECTION 2: Feature Engineering (5 Marks)
# ==========================================
# Create a new feature 'FamilySize' by combining SibSp (Siblings/Spouses) and Parch (Parents/Children)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Drop the old columns to avoid multicollinearity 
df = df.drop(['SibSp', 'Parch'], axis=1)

# ==========================================
# RUBRIC SECTION 3 & 4: Model Selection, Training & Eval (10 Marks)
# ==========================================
# Define features (X) and target (y)
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection Justification: 
# We choose Random Forest because it handles non-linear data well, is robust to outliers, 
# and provides feature importance which is useful for evaluation.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))