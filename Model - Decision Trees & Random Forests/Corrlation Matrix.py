import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Read the dataset
file_path = 'D:/Python/project/524/dataset\Dataset of Diabetes-V2.csv'
data = pd.read_csv(file_path)

# Select columns for analysis
columns = ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'CLASS']
selected_data = data[columns]

# Encode the gender column, convert F and M to numerical values
selected_data['Gender'] = selected_data['Gender'].map({'F': 0, 'M': 1})

# Encode the CLASS column, convert N and Y to numerical values
selected_data['CLASS'] = selected_data['CLASS'].map({'N': 0, 'Y': 1})

# Calculate the correlation matrix
correlation_matrix = selected_data.corr()

# Print the correlation matrix
print(correlation_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Medical Indicators')
plt.show()