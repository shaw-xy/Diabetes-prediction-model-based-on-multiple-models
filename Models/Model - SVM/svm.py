import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns

data = pd.read_csv('D:/Python/project/524/dataset\Dataset of Diabetes-V2.csv')
data['CLASS'] = data['CLASS'].str.strip()
#print(data['CLASS'].unique())

# 对Gender进进行标签编码
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender']) #F-0,M-1
print(data.head())

#将特征（X）和标签（y）分开
X = data.drop(['ID', 'CLASS'], axis=1)
y = data['CLASS']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#定义参数网格
param_grid = {
    'C': [0.1, 1, 10,100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']  # 核函数类型
}

# 使用网格搜索找到最佳参数
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=5)
grid.fit(X_train, y_train)

# 输出最佳参数
print(grid.best_params_)


# 使用最佳参数进行预测
y_pred = grid.predict(X_test)
y_train_pred = grid.predict(X_train)


# 评估模型
#混淆矩阵(test)
print("Confusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score (Test Set):")
print(accuracy_score(y_test, y_pred))


#混淆矩阵(train)
print("\nConfusion Matrix (Train Set):")
print(confusion_matrix(y_train, y_train_pred))

print("\nClassification Report (Train Set):")
print(classification_report(y_train, y_train_pred))

print("\nAccuracy Score (Train Set):")
print(accuracy_score(y_train, y_train_pred))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['N', 'Y'], yticklabels=['N', 'Y'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()