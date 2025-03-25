from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

# Load the dataset
data = pd.read_csv('D:/Python/project/524/dataset/Dataset of Diabetes .csv')

# Perform one-hot encoding on the 'Gender' column
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(data[['Gender']]).toarray()
column_names = encoder.get_feature_names_out(['Gender'])
encoded_df = pd.DataFrame(encoded_data, columns=column_names)
data = pd.concat([data.drop('Gender', axis=1), encoded_df], axis=1)
# print(data)

# Remove spaces from the values in the 'CLASS' column
data['CLASS'] = data['CLASS'].str.strip()

# Extract features and target variables
X = data.drop(['ID', 'No_Pation','CLASS'], axis=1)
y = data['CLASS']

# Strategy 1: Classify 'P' as 'Y'
y_strategy1 = y.replace('P', 'Y')

# Split the dataset into training and test sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y_strategy1, test_size=0.2, random_state=42)

# Train the model using logistic regression
model1 = LogisticRegression(max_iter=2000)
model1.fit(X_train1, y_train1)

# Make predictions and evaluate
y_test_pred1 = model1.predict(X_test1)
y_train_pred1 = model1.predict(X_train1)
accuracy_test_1 = accuracy_score(y_test1, y_test_pred1)
report_test_1 = classification_report(y_test1, y_test_pred1)

print("Strategy 1 ('P'->'Y')")
print("Accuracy of the test set('P'->'Y'):", accuracy_test_1)
print("Classification report:")
print(report_test_1)

# Training set
accuracy_train_1 = accuracy_score(y_train1, y_train_pred1)
report_train_1 = classification_report(y_train1, y_train_pred1)


print("Accuracy of training set ('P'->'Y'):", accuracy_train_1)
print("Classification report:")
print(report_train_1)


print('\n')


# Strategy 2: Classify 'P' as 'N'
y_strategy2 = y.replace('P', 'N')
# Split the dataset into training and test sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_strategy2, test_size=0.2, random_state=42)

# Build and train the logistic regression model
model2 = LogisticRegression(max_iter=2000)
model2.fit(X_train2, y_train2)

# Make predictions and evaluate
y_test_pred2 = model2.predict(X_test2)
y_train_pred2 = model2.predict(X_train2)

accuracy_test_2 = accuracy_score(y_test2, y_test_pred2)
report_test_2 = classification_report(y_test2, y_test_pred2)

# Training set
accuracy_train_2 = accuracy_score(y_train2, y_train_pred2)
report_train_2 = classification_report(y_train2, y_train_pred2)

print("Strategy 2 ('P' -> 'N'):")
print("Accuracy of test set ('P' -> 'N'):", accuracy_test_2)
print("Classification report:")
print(report_test_2)

print("Accuracy of training set ('P' -> 'N'):", accuracy_train_2)
print("Classification report:")
print(report_train_2)


print('\n')

# Strategy 3: Treat 'P' as a separate class
# Split the dataset into training and test sets
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the logistic regression model
model3 = LogisticRegression(max_iter=3000)
model3.fit(X_train3, y_train3)

# Make predictions and evaluate
y_test_pred3 = model3.predict(X_test3)
y_train_pred3 = model3.predict(X_train3)

accuracy_test_3 = accuracy_score(y_test3, y_test_pred3)
accuracy_train_3 = accuracy_score(y_train3, y_train_pred3)

report_test_3 = classification_report(y_test3, y_test_pred3)
report_train_3 = classification_report(y_train3, y_train_pred3)

print("Strategy 3 ('P' as a separate class) ")
print("Accuracy of the test set:", accuracy_test_3)
print("Classification report:")
print(report_test_3)


print("Accuracy of the training set('P' as a separate class):", accuracy_train_3)
print("Classification report:")
print(report_train_3)