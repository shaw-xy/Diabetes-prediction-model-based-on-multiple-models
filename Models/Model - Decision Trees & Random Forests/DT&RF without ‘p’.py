# Include glycated hemoglobin, exclude classification predicted as diabetes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Read the dataset
file_path = 'D:/Python/project/524/dataset\Dataset of Diabetes-V2.csv'
data = pd.read_csv(file_path)

# Remove the ID column
data = data.drop('ID', axis=1)

# Encode categorical variables
data = pd.get_dummies(data, columns=['Gender'])

# Split features and target variables
X = data.drop('CLASS', axis=1)
y = data['CLASS']

# Split the training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

# Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Output the classification report and confusion matrix for the Decision Tree
print("Decision Tree Classification Report:")
print(classification_report(y_test, dt_predictions))
print("Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, dt_predictions))

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=y.unique(), filled=True)
plt.show()

# Output the accuracy
print(f"Decision Tree Model Accuracy: {dt_accuracy}")

# Output the classification report and confusion matrix for the Random Forest
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

# Output the accuracy
print(f"Random Forest Model Accuracy: {rf_accuracy}")

# Calculate feature importances for the Decision Tree
dt_feature_importances = pd.Series(dt_model.feature_importances_, index=X.columns)
dt_feature_importances = dt_feature_importances.sort_values(ascending=False)

# Print the medical indicators more strongly correlated with diabetes (Decision Tree)
print("\nTop features related to diabetes according to Decision Tree:")
print(dt_feature_importances)

# Calculate feature importances for the Random Forest
rf_feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
rf_feature_importances = rf_feature_importances.sort_values(ascending=False)

# Print the medical indicators more strongly correlated with diabetes (Random Forest)
print("\nTop features related to diabetes according to Random Forest:")
print(rf_feature_importances)