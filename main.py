# ==========================================
# CA-2 PREDICTIVE ANALYTICS
# World Development Indicators Dataset
# ==========================================

# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==================================
# 2. Load Dataset
# ==================================

data = pd.read_csv("D:/fd/SEM-5/INT-234/CA2/P_Data_Extract_From_World_Development_Indicators/Data.csv")

# ==================================
# 3. Data Inspection
# ==================================


print("\nDataset Information:")
print(data.info())

print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns.tolist())

print("\nStatistical Summary:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Rows:")
print(data.duplicated().sum())

print("\nTotal Countries:")
print(data['Country Name'].nunique())

print("\nTotal Indicators:")
print(data['Series Name'].nunique())

# Identify Year Columns
year_cols = data.columns[4:]
print("\nYear Columns:")
print(year_cols.tolist())

# Check World Bank Missing Value Marker
print("\nMissing Marker Count:")
print((data[year_cols] == '..').sum())


# ==================================
# 4. Data Preprocessing
# ==================================


# Remove empty rows and footer rows
data = data.dropna(subset=['Country Name', 'Country Code',
                           'Series Name', 'Series Code'])

# Remove duplicate records
data = data.drop_duplicates()

# Select year columns from 2006 to 2024
year_cols = []
for col in data.columns:
    if col[:4].isdigit():
        year = int(col[:4])
        if year >= 2006 and year <= 2024:
            year_cols.append(col)

# Replace missing value symbol with NaN
data[year_cols] = data[year_cols].replace('..', np.nan)

# Convert year columns into numeric values
for col in year_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Convert wide dataset into long format
data_long = pd.melt(data,id_vars=['Country Name', 'Country Code', 'Series Name', 'Series Code'],
    value_vars=year_cols,var_name='Year',value_name='Value')

# Extract year from column name
data_long['Year'] = data_long['Year'].str[:4].astype(int)

# Convert long format into country-year format
df = data_long.pivot_table(index=['Country Name', 'Country Code', 'Year'],
                           columns='Series Name',values='Value')
df = df.reset_index()
df.columns.name = None

# Remove columns having more than 40% missing values
missing_percentage = df.isnull().mean() * 100
columns_to_keep = missing_percentage[missing_percentage <= 40].index.tolist()

# Keep target column
target = 'GDP growth (annual %)'
if target not in columns_to_keep:
    columns_to_keep.append(target)

df = df[columns_to_keep]

# Remove rows where target value is missing
df = df.dropna(subset=[target])

# Replace infinite values
df = df.replace([np.inf, -np.inf], np.nan)

print("\nAfter Data Preprocessing:")
print(df.head())

print("\nCleaned Dataset Shape:")
print(df.shape)

print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())


# ==================================
# 5. Exploratory Data Analysis (EDA)
# ==================================

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

print("\nAverage GDP Growth:")
print(df[target].mean())

print("\nMaximum GDP Growth:")
print(df[target].max())

print("\nMinimum GDP Growth:")
print(df[target].min())

print("\nGDP Growth by Year:")
print(df.groupby('Year')[target].mean())

print("\nTop 10 Countries by Average GDP Growth:")
country_growth = df.groupby('Country Name')[target].mean()

print(country_growth.sort_values(ascending=False).head(10))


# ==================================
# 6. Data Visualization
# ==================================


# GDP Growth Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df[target], bins=30, kde=True)
plt.title("Distribution of GDP Growth")
plt.xlabel("GDP Growth (Annual %)")
plt.ylabel("Frequency")
plt.grid()
plt.show()


# GDP Growth Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df[target])
plt.title("GDP Growth Boxplot")
plt.xlabel("GDP Growth (Annual %)")
plt.show()


# Correlation Heatmap
numeric_data = df.select_dtypes(include=np.number)

plt.figure(figsize=(14, 10))
sns.heatmap(numeric_data.corr(), cmap='coolwarm', annot=True, fmt='.2f')
plt.title("Correlation Heatmap")
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.show()

# Correlation of features with GDP Growth
correlation = numeric_data.corr()[target].drop(target)
correlation = correlation.sort_values()
plt.figure(figsize=(8, 6))
colors = ['#C0504D' if x < 0 else '#1F6E8C'
          for x in correlation]
plt.barh(correlation.index, correlation.values, color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.title("Correlation of Economic Indicators with GDP Growth")
plt.xlabel("Pearson Correlation")
plt.ylabel("Economic Indicators")
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
# Average GDP Growth by Year
year_growth = df.groupby('Year')[target].mean()

plt.figure(figsize=(10, 5))
plt.plot(year_growth.index, year_growth.values, marker='o')
plt.title("Average GDP Growth by Year")
plt.xlabel("Year")
plt.ylabel("Average GDP Growth")
plt.grid(True)
plt.show()


# Top 10 Countries by Average GDP Growth
country_growth = df.groupby('Country Name')[target].mean()

top_countries = country_growth.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
top_countries.plot(kind='barh')
plt.title("Top 10 Countries by Average GDP Growth")
plt.ylabel("Country")
plt.xlabel("Average GDP Growth")
plt.grid()
#plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ==================================
# 7. Feature Selection
# ==================================


# Select GDP Growth as target variable
target = 'GDP growth (annual %)'
X = df.drop(columns=['Country Name', 'Country Code', target])
y = df[target]
X = X.select_dtypes(include=np.number)

# Remove columns having no values
X = X.dropna(axis=1, how='all')

# Remove constant columns
X = X.loc[:, X.nunique() > 1]

print("\nIndependent Variables:")
print(X.head())

print("\nDependent Variable:")
print(y.head())

print("\nSelected Features:")
print(X.columns.tolist())

print("\nTotal Number of Features:")
print(X.shape[1])


# ==================================
# 8. Train-Test Split
# ==================================


# Split independent and dependent variables
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# Handle Missing Values
print("\nMissing Values Before Filling:")
print(X_train.isnull().sum().sum())

# Calculate median from training data
median_values = X_train.median()

# Replace missing values
X_train = X_train.fillna(median_values)
X_test = X_test.fillna(median_values)

# Remove columns which are still completely empty
X_train = X_train.dropna(axis=1, how='all')
X_test = X_test[X_train.columns]

print("\nMissing Values After Filling:")
print(X_train.isnull().sum().sum())
print(X_test.isnull().sum().sum())


# Normalization
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining Data After Normalization:")
print(X_train_scaled)

print("\nTesting Data After Normalization:")
print(X_test_scaled)


# ==================================
# 9. Linear Regression
# ==================================


model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Prediction
y_pred = model.predict(X_test_scaled)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nLinear Regression Results")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)
print("Mean Absolute Error (MAE):", mae)

plt.figure(figsize=(7, 5))

plt.scatter(y_test, y_pred, alpha=0.5, color='#009688')
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],'r--')
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Actual GDP Growth")
plt.ylabel("Predicted GDP Growth")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==================================
# 10. Decision Tree Regression
# ==================================


dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Prediction
dt_pred = dt_model.predict(X_test_scaled)

# Evaluation
dt_mse = mean_squared_error(y_test, dt_pred)
dt_rmse = np.sqrt(dt_mse)
dt_r2 = r2_score(y_test, dt_pred)
dt_mae = mean_absolute_error(y_test, dt_pred)

print("\nDecision Tree Regression Results")
print("Mean Squared Error (MSE):", dt_mse)
print("Root Mean Squared Error (RMSE):", dt_rmse)
print("R2 Score:", dt_r2)
print("Mean Absolute Error (MAE):", dt_mae)

# Actual vs Predicted Plot
plt.figure(figsize=(7, 5))

plt.scatter(y_test, dt_pred, alpha=0.5, color='#E67E22')
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],'r--')
plt.title("Decision Tree: Actual vs Predicted")
plt.xlabel("Actual GDP Growth")
plt.ylabel("Predicted GDP Growth")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==================================
# 11. Random Forest Regression
# ==================================


rf_model = RandomForestRegressor(
    n_estimators=100, random_state=42
)

rf_model.fit(X_train_scaled, y_train)

# Prediction
rf_pred = rf_model.predict(X_test_scaled)

# Evaluation
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)

print("\nRandom Forest Regression Results")
print("Mean Squared Error (MSE):", rf_mse)
print("Root Mean Squared Error (RMSE):", rf_rmse)
print("R2 Score:", rf_r2)
print("Mean Absolute Error (MAE):", rf_mae)

# Feature Importance
importance = rf_model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance', ascending=False
)

print("\nFeature Importance:")
print(feature_importance.head(10))

# Random Forest Test Error by Year
error = abs(y_test - rf_pred)
plt.figure(figsize=(8, 5))
sns.barplot(x=X_test['Year'], y=error, color='teal')
plt.title("Random Forest Test Error by Year")
plt.xlabel("Year")
plt.ylabel("Absolute Error")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 5))
plt.scatter(y_test, rf_pred, alpha=0.5, color='#8064A2')
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],'r--')
plt.title("Random Forest: Actual vs Predicted")
plt.xlabel("Actual GDP Growth")
plt.ylabel("Predicted GDP Growth")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==================================
# 12. KNN Regression
# ==================================


knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Prediction
knn_pred = knn_model.predict(X_test_scaled)

# Evaluation
knn_mse = mean_squared_error(y_test, knn_pred)
knn_rmse = np.sqrt(knn_mse)
knn_r2 = r2_score(y_test, knn_pred)
knn_mae = mean_absolute_error(y_test, knn_pred)

print("\nKNN Regression Results")
print("Mean Squared Error (MSE):", knn_mse)
print("Root Mean Squared Error (RMSE):", knn_rmse)
print("R2 Score:", knn_r2)
print("Mean Absolute Error (MAE):", knn_mae)

plt.figure(figsize=(7, 5))
plt.scatter(y_test, knn_pred, alpha=0.5, color='#D65F5F')
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],'r--')

plt.title("KNN: Actual vs Predicted")
plt.xlabel("Actual GDP Growth")
plt.ylabel("Predicted GDP Growth")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==================================
# 13. Support Vector Regression (SVM)
# ==================================


svm_model = SVR(kernel='rbf')
svm_model.fit(X_train_scaled, y_train)

# Prediction
svm_pred = svm_model.predict(X_test_scaled)

# Evaluation
svm_mse = mean_squared_error(y_test, svm_pred)
svm_rmse = np.sqrt(svm_mse)
svm_r2 = r2_score(y_test, svm_pred)
svm_mae = mean_absolute_error(y_test, svm_pred)

print("\nSupport Vector Regression Results")
print("Mean Squared Error (MSE):", svm_mse)
print("Root Mean Squared Error (RMSE):", svm_rmse)
print("R2 Score:", svm_r2)
print("Mean Absolute Error (MAE):", svm_mae)

plt.figure(figsize=(7, 5))

plt.scatter(y_test, svm_pred, alpha=0.5, color='#70AD47')
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],'r--')
plt.title("SVR: Actual vs Predicted")
plt.xlabel("Actual GDP Growth")
plt.ylabel("Predicted GDP Growth")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==================================
# 14. Model Performance Comparison
# ==================================


results = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'KNN', 'SVM'],
    'MAE': [mae, dt_mae, rf_mae, knn_mae, svm_mae],
    'MSE': [mse, dt_mse, rf_mse, knn_mse, svm_mse],
    'RMSE': [rmse, dt_rmse, rf_rmse, knn_rmse, svm_rmse],
    'R2 Score': [r2, dt_r2, rf_r2, knn_r2, svm_r2]
})

print(results)

# Model Comparison - Actual vs Predicted

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.scatter(y_test,y_pred, color='teal', alpha=0.5)
plt.title("Linear Regression")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid()

plt.subplot(2, 3, 2)
plt.scatter(y_test, dt_pred, color='orange', alpha=0.5)
plt.title("Decision Tree")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid()

plt.subplot(2, 3, 3)
plt.scatter(y_test, rf_pred, color='green', alpha=0.5)
plt.title("Random Forest")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid()

plt.subplot(2, 3, 4)
plt.scatter(y_test, knn_pred, color='purple', alpha=0.5)
plt.title("KNN")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid()

plt.subplot(2, 3, 5)
plt.scatter(y_test, svm_pred, color='brown', alpha=0.5)
plt.title("SVR")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid()

plt.tight_layout()
plt.show()


# ==================================
# 15. Final Conclusion
# ==================================


best_model = results.loc[results['R2 Score'].idxmax()]
print("Best Performing Model:", best_model['Model'])
print("Best R2 Score:", best_model['R2 Score'])
print("The project uses World Bank indicators to estimate GDP growth.")
print("Five machine learning models were implemented and compared.")
print("Model performance was evaluated using MAE, MSE, RMSE and R2 Score.")