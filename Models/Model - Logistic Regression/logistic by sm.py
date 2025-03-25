from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv('D:/Python/project/524/dataset\Dataset of Diabetes-V2.csv')

# 对 Gender 列进行独热编码
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(data[['Gender']]).toarray()
column_names = encoder.get_feature_names_out(['Gender'])
encoded_df = pd.DataFrame(encoded_data, columns=column_names)
data = pd.concat([data.drop('Gender', axis=1), encoded_df], axis=1)

# 去除 CLASS 列值中的空格
data['CLASS'] = data['CLASS'].str.strip()

# 对 CLASS 列进行编码
le = LabelEncoder()
data['CLASS'] = le.fit_transform(data['CLASS'])

# 划分数据集
X = data.drop(['ID', 'CLASS'], axis=1)
y = data['CLASS']

# 8/2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7/3
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# model2 by sm

#添加常数项
X_train = sm.add_constant(X_train)

#构建和拟合逻辑回归模型
model = sm.GLM(y_train, X_train, family=sm.families.Binomial(sm.families.links.logit()))
results = model.fit()

print("summary")
print(results.summary())