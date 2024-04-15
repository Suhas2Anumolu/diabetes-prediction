from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# Sample dataset
data = {
    'glucose': [100, 150, 120, 80, 95, 130, 140, 170, 110, 105],
    'blood_pressure': [70, 80, 90, 60, 75, 85, 95, 100, 65, 70],
    'diabetes': [0, 1, 1, 0, 0, 1, 1, 1, 0, 0]  # 0: Not diabetic, 1: Diabetic
}

df = pd.DataFrame(data)

# Split dataset into features (X) and target variable (y)
X = df[['glucose', 'blood_pressure']]
y = df['diabetes']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
