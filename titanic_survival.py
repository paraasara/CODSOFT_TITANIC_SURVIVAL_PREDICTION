# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the dataset
# Replace 'your_dataset.csv' with the path to your dataset
data = pd.read_csv('c:\\Users\ABC\Desktop\codsoft\Titanic-Dataset.csv')

# Step 2: Explore and preprocess the data
print(data.info())  # To understand the structure of the data
print(data.describe())  # To understand statistical details of numerical features

# Handle missing values
# Example: Fill missing 'Age' with the median and 'Embarked' with the mode
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop columns that might not be useful for prediction
# Example: 'PassengerId', 'Name', 'Ticket', 'Cabin'
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Encode categorical features
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])  # Male = 1, Female = 0
data['Embarked'] = le.fit_transform(data['Embarked'])  # Encode ports numerically

# Separate features (X) and target (y)
X = data.drop('Survived', axis=1)
y = data['Survived']

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Build and train the model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Predict for new data
# Example input (same features as X): [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
new_passenger = np.array([[3, 1, 22, 1, 0, 7.25, 1]])  # Example data
new_passenger_scaled = scaler.transform(new_passenger)
prediction = model.predict(new_passenger_scaled)
print("Survived (1 = Yes, 0 = No):", prediction[0])
