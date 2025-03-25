# Diabetes Prediction System | 糖尿病预测系统

A comprehensive machine learning-based diabetes prediction system using multiple models and ensemble methods.
基于多个机器学习模型和集成方法的综合糖尿病预测系统。

## Project Overview | 项目概述

This project implements a diabetes prediction system that analyzes patient information using various machine learning models to predict the probability of diabetes. The system includes multiple individual models and an ensemble approach for improved accuracy.

本项目实现了一个糖尿病预测系统，通过多个机器学习模型分析患者信息来预测糖尿病的概率。系统包含多个独立模型和集成方法以提高预测准确性。

## Project Structure | 项目结构

```
524/
├── dataset/                    # Dataset files and preprocessing
│   ├── Dataset of Diabetes.csv     # Original dataset
│   └── Dataset of Diabetes-V2.csv  # Processed dataset
│
├── Model - Logistic Regression/    # Logistic Regression implementation
│
├── Model - Decision Trees & Random Forests/  # Tree-based models
│
├── Model - SVM/                    # Support Vector Machine implementation
│
├── Voting - Ensemble/              # Ensemble learning implementation
│
└── Ui/                            # User Interface implementation
    └── medical-platform/          # Medical platform web application
```

## Models | 模型

The project implements four different approaches:

1. **Logistic Regression | 逻辑回归**
   - Basic probabilistic classification
   - Feature relationship analysis
   - Baseline model performance

2. **Decision Trees & Random Forests | 决策树和随机森林**
   - Decision Tree for interpretable results
   - Random Forest for improved accuracy
   - Feature importance analysis

3. **Support Vector Machine (SVM) | 支持向量机**
   - Non-linear classification
   - Kernel-based approach
   - High-dimensional data handling

4. **Ensemble Method | 集成方法**
   - Stacking approach
   - Combines predictions from all models
   - Improved overall accuracy

## Dataset | 数据集

Source: [Diabetes Dataset on Mendeley](https://data.mendeley.com/datasets/wj9rwkp9c2/1)

The dataset includes the following features:
- Gender
- Age
- Urea
- Cr
- HbA1c
- Chol
- TG
- HDL
- LDL
- VLDL
- BMI

Two versions of the dataset are included:
- `Dataset of Diabetes.csv`: Original dataset
- `Dataset of Diabetes-V2.csv`: Processed and standardized dataset

## Technical Requirements | 技术要求

### Backend (Model Development) | 后端（模型开发）
- Python
- Required libraries:
  - scikit-learn
  - numpy
  - pandas
  - matplotlib
  - seaborn

### Frontend (UI) | 前端（界面）
- Python Flask
- HTML/CSS/JavaScript
- SQLite

## Installation & Setup | 安装和设置

1. Clone the repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Navigate to the UI directory:
   ```bash
   cd Ui/medical-platform
   ```
4. Start the application:
   ```bash
   python app.py
   ```

## Usage Guide | 使用指南

1. Access the web interface through your browser
2. Input patient data including:
   - Personal information
   - Medical test results
   - Physical measurements
3. Select the desired prediction model
4. View the prediction results and analysis

## Model Performance | 模型性能

Each model has been evaluated using:
- Cross-validation
- Accuracy metrics
- ROC curves
- Confusion matrices

The ensemble method typically provides the best overall performance by combining the strengths of individual models.

## Contributing | 贡献指南

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License | 许可证

This project is licensed under the MIT License.
本项目采用 MIT 许可证。

## Acknowledgments | 致谢

- Dataset provided by Mendeley Data
- Contributors and researchers in the field of medical data analysis
- Open source community for various tools and libraries used

## Contact | 联系方式

For questions and support, please open an issue in the repository.
如有问题和支持需求，请在仓库中创建新的 issue。 