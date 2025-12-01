## Day 1

### Colab Link

https://colab.research.google.com/drive/15qcte1DpYRwLeaaIJXAsHw0JDXbIzGGw?usp=sharing



<img width="1742" height="1152" alt="image" src="https://github.com/user-attachments/assets/eab9857c-a56e-48dc-9386-bf80ac90d56c" />


<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/32fa692f-9f21-4431-9db0-06437ad56ef4" />

# Data Processing Steps

## 1. Load Data
```python
df = pd.read_excel("/content/rental_car_data_with_eda_issues.xlsx")
```

# Code Actions Reference

| Code Action            | Logic / Description                                                                                     |
|------------------------|-----------------------------------------------------------------------------------------------------------|
| `pd.read_excel(...)`   | Loads the data into a pandas DataFrame (`df`).                                                            |
| `missingno.bar(df)`    | Visualizes the distribution of non-missing data across all columns to quickly identify fields with missing values. |

```

# Imputation Strategy

| Column              | Imputation Method     | Rationale                                                                                     |
|---------------------|------------------------|-------------------------------------------------------------------------------------------------|
| `Customer_Age`      | Median (`.median()`)   | Median is robust to potential outliers in age data.                                            |
| `Fuel_Used`         | Mean (`.mean()`)       | Mean is used for continuous, ratio-scaled data, assuming a symmetric distribution.             |
| `Distance_Travelled`| Median (`.median()`)   | Median is robust to potential outliers (e.g., extremely long or short trips).                  |

```

# Outlier Detection Using IQR

| Step                | Calculation / Action                                                                                                                                      | Description                                                                                                                                                                                                                   |
|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Define Quartiles    | ```python\nQ1 = df['Rental_Duration'].quantile(0.25)\nQ3 = df['Rental_Duration'].quantile(0.75)\n```                                                      | Calculates the 25th (Q1) and 75th (Q3) percentiles.                                                                                                                                                                           |
| Calculate IQR       | ```python\nIQR = Q3 - Q1\n```                                                                                                                             | Measures the spread of the middle 50% of the data.                                                                                                                                                                            |
| Set Bounds          | ```python\nlower_bound = Q1 - 1.5 * IQR\nupper_bound = Q3 + 1.5 * IQR\n```                                                                               | Defines the boundaries beyond which data points are considered outliers.                                                                                                                                                      |
| Filter Data         | ```python\ndf_no_outliers = df[(df['Rental_Duration'] >= lower_bound) & (df['Rental_Duration'] <= upper_bound)]\n```                                      | Creates a new DataFrame free of duration outliers. **Note:** Subsequent steps reuse the original `df`, meaning this filtering step was for analysis/demo only, or `df` may have been overwritten later in the workflow. |


## 5. Data Splitting and Scaling ⚖️

The dataset is prepared for modeling by defining features and the target variable, splitting the data, and scaling numerical features.

### **Define X and y**
- **Target (`y`)**: `Rental_Cost`  
- **Features (`X`)**: All remaining columns

```python
X = df.drop('Rental_Cost', axis=1)
y = df['Rental_Cost']


## 6. Model Training and Evaluation 🎯

A Linear Regression model is trained and evaluated using standard regression metrics to assess its predictive performance.

### **Step-by-Step Process**

| Step               | Code Logic                                                                 | Description                                                                                      |
|-------------------|----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| **Data Preparation** | ```python\nX_train_numeric = X_train.drop(columns=['Rental_ID', 'Date'])\nX_test_numeric = X_test.drop(columns=['Rental_ID', 'Date'])``` | Non-numeric or non-predictive columns are removed before training.                              |
| **Training**         | ```python\nlr_model.fit(X_train_numeric, y_train)\n```                    | The Linear Regression model is trained using the cleaned training data.                         |
| **Prediction**       | ```python\nlr_pred = lr_model.predict(X_test_numeric)\n```                | The trained model generates predictions on the unseen test set.                                 |
| **Evaluation**       | ```python\nfrom sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n\nmae = mean_absolute_error(y_test, lr_pred)\nmse = mean_squared_error(y_test, lr_pred)\nr2 = r2_score(y_test, lr_pred)\n``` | Model performance is assessed using **MAE**, **MSE**, and **R² Score**.                         |

---

### **Summary of Metrics**
- **MAE (Mean Absolute Error):** Measures average prediction error in absolute terms.  
- **MSE (Mean Squared Error):** Penalizes larger errors more heavily.  
- **R² Score:** Indicates how well the model explains variance in the target variable.  


