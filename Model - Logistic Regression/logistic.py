from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd



data = pd.read_csv('D:/Python/project/524/dataset\Dataset of Diabetes-V2.csv')

# do one-hot to "Gender"
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(data[['Gender']]).toarray()
column_names = encoder.get_feature_names_out(['Gender'])
encoded_df = pd.DataFrame(encoded_data, columns=column_names)
data = pd.concat([data.drop('Gender', axis=1), encoded_df], axis=1)
# print(data)

# Remove spaces from CLASS column values
data['CLASS'] = data['CLASS'].str.strip()
# Encoding the CLASS column
# le = LabelEncoder()
# data['CLASS'] = le.fit_transform(data['CLASS'])
#

# divide datase
X = data.drop(['ID', 'CLASS'], axis=1)
y = data['CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("result summary:")

#Make predictions on the test set
y_test_pred = model.predict(X_test)

#Calculate the accuracy of the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Accuracy of test set: {test_accuracy * 100:.2f}%')

# output report
print('test set classification report:')
print(classification_report(y_test, y_test_pred))

print('\n')

# Make predictions on the training set
y_train_pred = model.predict(X_train)

#Calculate the accuracy of the training set
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f'Accuracy of training set: {train_accuracy * 100:.2f}%')

# output report
print('Training set classification report:')
print(classification_report(y_train, y_train_pred))



