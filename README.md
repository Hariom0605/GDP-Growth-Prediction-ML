# GDP Growth Prediction Using Machine Learning

## Overview

GDP growth is an important indicator of a country's economic performance. It is influenced by several factors, including inflation, foreign investment, population, industrial development, and services.
This project explores how machine learning can be used to estimate GDP growth using economic indicators obtained from the World Bank's World Development Indicators (WDI) dataset.
The project covers data preprocessing, exploratory data analysis, regression model development, performance evaluation, and feature importance analysis.

## Objectives
- Analyze historical economic indicators from the World Bank.
- Clean and preprocess real-world economic data.
- Explore relationships between economic indicators and GDP growth.
- Implement and compare multiple regression algorithms.
- Identify the best-performing model.
- Understand the contribution of different features through feature importance analysis.

## Dataset
**Source:** World Bank – World Development Indicators
**Time Period:** 2006–2024
**Dataset Summary:**
- Original dataset: 5,857 rows and 24 columns.
- Final processed dataset: 4,869 observations.
- Reporting entities: 262.
- Predictor features: 21.
- Training observations: 3,895.
- Testing observations: 974.

**Target Variable:** GDP Growth (Annual %)

Some of the selected indicators include:
- Inflation
- Foreign Direct Investment
- Industry Value Added
- Services Value Added
- Population
- Renewable Energy Consumption
- Mobile Subscriptions
- Urban Population
- Health Expenditure

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Data Preprocessing

The following preprocessing operations were performed:

1. Dataset inspection and cleaning.
2. Handling missing values.
3. Conversion of economic indicators into numerical format.
4. Transformation from wide format to country-year format.
5. Removal of features with more than 40% missing values.
6. Median imputation for remaining missing predictor values.
7. Min-Max feature scaling.
8. Training and testing data split.

## Machine Learning Models

Five regression algorithms were implemented:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. K-Nearest Neighbors Regressor
5. Support Vector Regression (SVR)

## Model Evaluation

The models were evaluated using four regression metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

### Results

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 3.3950 | 42.1211 | 6.4901 | 0.0503 |
| Decision Tree | 3.1547 | 36.5211 | 6.0433 | 0.1765 |
| Random Forest | **2.5619** | **34.0425** | **5.8346** | **0.2324** |
| KNN | 3.3548 | 46.5090 | 6.8197 | -0.0487 |
| SVR | 3.2482 | 41.7762 | 6.4635 | 0.0580 |

### Best Performing Model

Random Forest achieved the best overall performance among the five evaluated models.

- MAE: 2.5619
- MSE: 34.0425
- RMSE: 5.8346
- R²: 0.2324

## Visualizations

The project includes visual analysis of:

- GDP Growth Distribution
- Correlation Heatmap
- Economic Indicator Correlation
- Model Performance Comparison
- Actual vs Predicted GDP Growth
- Random Forest Feature Importance
- Year-wise Random Forest Prediction Error

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/GDP-Growth-Prediction-ML.git
```

### 2. Open the project folder

```bash
cd GDP-Growth-Prediction-ML
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Python file

```bash
python main.py
```

Make sure the dataset is placed inside the `dataset` folder and the file path in the Python code matches its location.

## Important Note

This project performs same-year GDP growth estimation using economic indicators from the same reporting year. It should not be interpreted as a future-year forecasting system.

The model results represent one experimental train-test split. Further improvements can be explored through hyperparameter tuning, time-based validation, and the use of lagged economic indicators.

## Future Scope

- Experiment with additional economic indicators.
- Improve model performance through hyperparameter tuning.
- Apply time-based and entity-based validation.
- Explore genuine future GDP forecasting using previous-year indicators.
- Develop an interactive dashboard for exploring predictions.

## Author

**Hariom Tiwari**

B.Tech – Computer Science and Engineering  
Lovely Professional University

## Acknowledgement

Data used in this project was obtained from the World Bank's World Development Indicators database.

Dataset: https://databank.worldbank.org/source/world-development-indicators
