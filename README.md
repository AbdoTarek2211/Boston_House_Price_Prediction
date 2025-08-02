# Boston House Price Prediction

A comprehensive machine learning project for predicting Boston housing prices using various regression algorithms and data preprocessing techniques.

## 📊 Project Overview

This project implements a complete machine learning pipeline for Boston house price prediction, exploring different data preprocessing methods, feature selection techniques, and regression algorithms to achieve optimal prediction accuracy.

### 🎯 Objectives
- Perform comprehensive exploratory data analysis on the Boston housing dataset
- Compare different outlier handling strategies (removal vs capping)
- Evaluate various feature scaling methods (StandardScaler vs RobustScaler)
- Implement feature selection to identify the most predictive variables
- Build and optimize multiple regression models
- Provide detailed performance comparison across different approaches

## 📁 Dataset

The project uses the Boston Housing dataset, which contains information about housing in the area of Boston, Massachusetts.

**Features include:**
- `CRIM`: Per capita crime rate by town
- `ZN`: Proportion of residential land zoned for lots over 25,000 sq.ft
- `INDUS`: Proportion of non-retail business acres per town
- `CHAS`: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- `NOX`: Nitric oxides concentration (parts per 10 million)
- `RM`: Average number of rooms per dwelling
- `AGE`: Proportion of owner-occupied units built prior to 1940
- `DIS`: Weighted distances to employment centers
- `RAD`: Index of accessibility to radial highways
- `TAX`: Full-value property-tax rate per $10,000
- `PTRATIO`: Pupil-teacher ratio by town
- `B`: Proportion of blacks by town
- `LSTAT`: % lower status of the population
- `MEDV`: Median value of owner-occupied homes in $1000's (target variable)

## 🛠️ Technologies Used

- **Python 3.x**
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Models**: Linear Regression, Ridge, Lasso, Random Forest, SVR

## 📋 Requirements

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
warnings
```

## 🚀 Installation & Usage

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/boston-house-price-prediction.git
cd boston-house-price-prediction
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Prepare your data**
   - Ensure you have the `boston.csv` file in your working directory
   - Update the file path in the code if necessary

4. **Run the analysis**
```bash
python boston_house_price_prediction.py
```

## 📊 Analysis Workflow

### 1. **Data Exploration & Quality Assessment**
- Dataset information and statistical summary
- Missing values and duplicate detection
- Feature correlation analysis

### 2. **Exploratory Data Analysis**
- Correlation heatmaps for feature relationships
- Distribution plots for all variables
- Boxplots for outlier identification

### 3. **Data Preprocessing**

#### Outlier Handling Strategies:
- **Method 1**: Complete outlier removal using IQR method
- **Method 2**: Outlier capping (winsorization) to boundary values

#### Feature Scaling Comparison:
- **StandardScaler**: Standardizes features by removing mean and scaling to unit variance
- **RobustScaler**: Uses median and IQR, more robust to outliers

### 4. **Feature Selection**
- SelectKBest with f_classif scoring
- Identification of top 5 most predictive features
- Correlation analysis of selected features

### 5. **Model Development & Evaluation**

#### Baseline Models:
- Linear Regression
- Support Vector Regression (Linear kernel)
- Random Forest Regressor

#### Hyperparameter Optimization:
- **Ridge Regression**: GridSearchCV for alpha tuning
- **Lasso Regression**: Manual cross-validation for alpha selection
- **Random Forest**: Comprehensive grid search across multiple parameters

### 6. **Model Validation**
- 5-fold cross-validation for all models
- Train-test split (70-30) for final evaluation
- Performance metrics: MSE and R²

## 📈 Key Results

The analysis provides comprehensive performance comparisons across:
- Different data preprocessing approaches
- Various feature selection strategies
- Multiple regression algorithms with optimal hyperparameters

### Performance Metrics
- **Mean Squared Error (MSE)**: Lower values indicate better performance
- **R-squared (R²)**: Higher values indicate better model fit (closer to 1.0)

## 🔍 Key Insights

1. **Outlier Treatment**: Comparison between removal vs capping strategies shows their impact on model performance
2. **Feature Scaling**: RobustScaler generally performs better with outlier-prone data
3. **Feature Selection**: Top 5 features provide competitive performance with reduced complexity
4. **Model Performance**: Hyperparameter tuning significantly improves baseline model performance

## 📊 Visualizations

The project generates multiple visualizations:
- Correlation matrices before/after preprocessing
- Feature distribution plots
- Boxplots for outlier analysis
- Scaling method comparisons

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Boston Housing dataset from the UCI Machine Learning Repository
- Scikit-learn community for excellent machine learning tools
- Matplotlib and Seaborn for powerful visualization capabilities
---

⭐ **If you found this project helpful, please give it a star!** ⭐
