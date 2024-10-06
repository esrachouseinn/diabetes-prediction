import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, \
    f1_score

# Load the dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Display the first few rows of the dataset
print(diabetes_dataset.head())

# Display dataset statistics
print(diabetes_dataset.describe())

# Display class distribution in the Outcome column
print(diabetes_dataset['Outcome'].value_counts())

# Group by Outcome and display the mean of each feature
print(diabetes_dataset.groupby('Outcome').mean())

# Check for missing data
print(diabetes_dataset.isnull().sum())

# Data Visualization: Outcome Distribution
sns.countplot(x='Outcome', data=diabetes_dataset, hue='Outcome', palette="viridis", legend=False)
plt.title("Outcome Distribution")
plt.show()

# Data Visualization: Feature Distribution by Outcome
plt.figure(figsize=(12, 10))
for idx, column in enumerate(diabetes_dataset.columns[:-1], 1):
    plt.subplot(4, 2, idx)
    sns.histplot(data=diabetes_dataset, x=column, hue='Outcome', kde=True, palette="coolwarm")
    plt.title(f"{column} Distribution by Outcome")
plt.tight_layout()
plt.show()

# Correlation Matrix Visualization
corr_matrix = diabetes_dataset.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Feature and target separation
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardizing the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# 1. Model: SVM with Hyperparameter Tuning using GridSearchCV
svm_model = svm.SVC()
params = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}
svm_grid = GridSearchCV(svm_model, param_grid=params, cv=5)
svm_grid.fit(X_train, Y_train)

# Best SVM model after tuning
best_svm_model = svm_grid.best_estimator_
print(f"Best SVM Parameters: {svm_grid.best_params_}")

# 2. Model: Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, Y_train)

# 3. Model: Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, Y_train)

# 4. Model: Neural Network (MLPClassifier)
mlp_model = MLPClassifier(max_iter=1000)
mlp_model.fit(X_train, Y_train)

# Compare models on training and test data
models = {
    'SVM': best_svm_model,
    'Logistic Regression': logistic_model,
    'Random Forest': rf_model,
    'Neural Network': mlp_model
}

# Evaluating model performance
for model_name, model in models.items():
    # Training accuracy
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(Y_train, train_predictions)

    # Test accuracy
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(Y_test, test_predictions)

    # Print the results
    print(f"\n{model_name} - Training Accuracy: {train_accuracy:.4f}")
    print(f"{model_name} - Test Accuracy: {test_accuracy:.4f}")

    # Precision, Recall, F1 Score
    precision = precision_score(Y_test, test_predictions)
    recall = recall_score(Y_test, test_predictions)
    f1 = f1_score(Y_test, test_predictions)

    print(f"{model_name} - Precision: {precision:.4f}")
    print(f"{model_name} - Recall: {recall:.4f}")
    print(f"{model_name} - F1 Score: {f1:.4f}")

    # Confusion Matrix and Classification Report
    print(f"\n{model_name} Classification Report:")
    print(classification_report(Y_test, test_predictions))
    print(f"Confusion Matrix:\n{confusion_matrix(Y_test, test_predictions)}")

    # Confusion Matrix Visualization
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(Y_test, test_predictions), annot=True, cmap="Blues", fmt="d", cbar=False)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Check for overfitting or underfitting
    print(f'{model_name} - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
    if train_accuracy > test_accuracy:
        print(f'{model_name} is likely overfitting.')
    elif train_accuracy < test_accuracy:
        print(f'{model_name} is likely underfitting.')
    else:
        print(f'{model_name} is well-fitted.')

# Make predictions on new data using the best SVM model
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
std_data = scaler.transform(input_data_as_numpy_array)

# Predict and print the result
prediction = best_svm_model.predict(std_data)
if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")
