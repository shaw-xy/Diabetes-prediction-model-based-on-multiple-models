import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

#数据预处理：
data = pd.read_csv('D:/Python/project/524/dataset\Dataset of Diabetes-V2.csv')
data['CLASS'] = data['CLASS'].str.strip()
#print(data['CLASS'].unique())

# 对Gender进行独热编码
data = pd.get_dummies(data, columns=['Gender'])
# print(data.head())

#将特征（X）和标签（y）分开
X = data.drop(['ID', 'CLASS'], axis=1)
y = data['CLASS']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#1.SVM模型训练
#调整svm的超参数
param_grid = {
    'C': [0.1, 1, 10,100],  # 惩罚参数，值越小，允许的间隔越大，模型越简单
    'gamma': [0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# 网格搜索
grid = GridSearchCV(SVC( probability=True), param_grid, refit=True, verbose=0)
grid.fit(X_train, y_train)

# 输出最佳参数
#print(grid.best_params_)

# 使用最佳参数进行预测
y_pred = grid.predict(X_test)
y_train_pred = grid.predict(X_train)


#2.逻辑回归模型训练
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)


#3.随机森林模型训练
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_train_predictions = rf_model.predict(X_train)
#rf_accuracy = accuracy_score(y_test, rf_predictions)


# 提取训练好的模型
logistic_model = model  # 逻辑回归模型
svm_model = grid.best_estimator_  # 支持向量机模型
rf_model = rf_model  # 随机森林模型

'''
# 创建硬投票集成模型
hard_voting_clf = VotingClassifier(
    estimators=[
        ('lr', logistic_model),  # 使用训练好的逻辑回归模型
        ('svm', svm_model),      # 使用训练好的支持向量机模型
        ('rf', rf_model)         # 使用训练好的随机森林模型
    ],
    voting='hard'  # 硬投票
)
'''
# 创建软投票集成模型
soft_voting_clf = VotingClassifier(
    estimators=[
        ('lr', logistic_model),  # 使用训练好的逻辑回归模型
        ('svm', svm_model),      # 使用训练好的支持向量机模型
        ('rf', rf_model)         # 使用训练好的随机森林模型
    ],
    voting='soft'  # 软投票
)



'''
# 硬投票模型预测
hard_voting_clf.fit(X_train, y_train)  
y_pred_hard = hard_voting_clf.predict(X_test)
hard_accuracy = accuracy_score(y_test, y_pred_hard)
y_train_pred_hard = hard_voting_clf.predict(X_train)
train_accuracy_hard = accuracy_score(y_train, y_train_pred_hard)
'''

# 软投票模型预测
soft_voting_clf.fit(X_train, y_train)  #即使模型已训练，也需要调用 fit
soft_voting_clf.estimators_ = [logistic_model,svm_model,rf_model] #将 VotingClassifier 的 estimators_ 属性替换为已经训练好的模型


y_pred_soft = soft_voting_clf.predict(X_test)
soft_accuracy = accuracy_score(y_test, y_pred_soft)
y_train_pred_soft = soft_voting_clf.predict(X_train)
train_accuracy_soft = accuracy_score(y_train, y_train_pred_soft)

'''
# 输出结果
print("Hard Voting Results:")
print(f"Test Accuracy: {hard_accuracy * 100:.2f}%")
print("Test Classification Report:")
print(classification_report(y_test, y_pred_hard))
print("Hard Voting Results (Training Set):")
print(f"Train Accuracy: {train_accuracy_hard * 100:.2f}%")
print("Train Classification Report:")
print(classification_report(y_train, y_train_pred_hard))
'''
print("\nSoft Voting Results:(Test set)")
print(f"Test Accuracy: {soft_accuracy * 100:.2f}%")
print("Test Classification Report:")
print(classification_report(y_test, y_pred_soft))
print("\nSoft Voting Results (Train Set):")
print(f"Train Accuracy: {train_accuracy_soft * 100:.2f}%")
print("Train Classification Report:")
print(classification_report(y_train, y_train_pred_soft))
