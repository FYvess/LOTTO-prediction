# Lotto Number Prediction using historical data with RandomForestRegressor and GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
# Assuming we have historical lotto data
data = {
    'DrawDate': pd.to_datetime([
        '2025-02-08', '2025-02-06', '2025-02-04', '2025-02-01',
        '2025-01-30', '2025-01-28', '2025-01-25', '2025-01-23', '2025-01-21',
        '2025-01-18', '2025-01-16', '2025-01-14', '2025-01-11', '2025-01-09',
        '2025-01-07', '2025-01-04', 
    ]),
    'WinningNumber1': [28, 36, 33, 8, 42, 10, 10, 20, 29, 9, 32, 16, 9, 25, 25, 24],
    'WinningNumber2': [34, 6, 19, 29, 4, 22, 12, 27, 19, 13, 25, 12, 38, 4, 37, 41],
    'WinningNumber3': [42, 13, 29, 18, 22, 31, 26, 40, 35, 40, 38, 31, 23, 16, 9, 2],
    'WinningNumber4': [5, 12, 21, 15, 28, 28, 15, 16, 30, 42, 42, 34, 12, 32, 41, 36],
    'WinningNumber5': [17, 28, 4, 38, 31, 11, 27, 19, 12, 14, 40, 3, 32, 22, 20, 6],
    'WinningNumber6': [41, 37, 22, 27, 12, 14, 11, 12, 15, 33, 34, 37, 29, 34, 34, 5]
}
# Convert dataset into DataFrame
df = pd.DataFrame(data)

# Clip the winning numbers to be within the range 1 to 42
df[['WinningNumber1', 'WinningNumber2', 'WinningNumber3', 'WinningNumber4', 'WinningNumber5', 'WinningNumber6']] = df[['WinningNumber1', 'WinningNumber2', 'WinningNumber3', 'WinningNumber4', 'WinningNumber5', 'WinningNumber6']].clip(upper=42)

# Ensure the numbers are unique for each row
def make_unique(row):
    unique_numbers = list(set(row))
    while len(unique_numbers) < 6:
        unique_numbers.append(np.random.randint(1, 43))
    np.random.shuffle(unique_numbers)
    return [int(num) for num in unique_numbers]

df[['WinningNumber1', 'WinningNumber2', 'WinningNumber3', 'WinningNumber4', 'WinningNumber5', 'WinningNumber6']] = df[['WinningNumber1', 'WinningNumber2', 'WinningNumber3', 'WinningNumber4', 'WinningNumber5', 'WinningNumber6']].apply(make_unique, axis=1, result_type='expand')

print("Lotto draw data:")
print(df.head())

# Use the winning numbers as the target variables
X = df[['DrawDate']]
y = df[['WinningNumber1', 'WinningNumber2', 'WinningNumber3', 'WinningNumber4', 'WinningNumber5', 'WinningNumber6']]

# Convert dates to ordinal for model training
X['DrawDate'] = X['DrawDate'].map(pd.Timestamp.toordinal)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize the RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions using the testing set
y_pred = best_model.predict(X_test)

# Clip the predicted numbers to be within the range 1 to 42
y_pred = np.clip(y_pred, 1, 42)

# Round the predicted numbers and convert to integers
y_pred = np.round(y_pred).astype(int)

# Ensure the predicted numbers are unique for each row
y_pred = np.apply_along_axis(make_unique, 1, y_pred)

# Calculate the mean squared error and r2 score for each winning number
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
r2 = r2_score(y_test, y_pred, multioutput='raw_values')

# Round the mse and r2 values to 2 decimal places
mse = np.round(mse, 2)
r2 = np.round(r2, 2)

print("Mean Squared Error for each winning number: ", mse)
print("R2 Score for each winning number: ", r2)

# Convert ordinal dates back to datetime for plotting
X_test['DrawDate'] = X_test['DrawDate'].map(pd.Timestamp.fromordinal)

# Plot the actual numbers and predicted numbers
plt.figure(figsize=(10, 6))
for i in range(1, 7):
    plt.plot(X_test['DrawDate'], y_test.iloc[:, i-1], label=f'Actual Winning Number {i}', marker='o')
    plt.plot(X_test['DrawDate'], y_pred[:, i-1], label=f'Predicted Winning Number {i}', marker='x')
plt.xlabel('Draw Date')
plt.ylabel('Winning Number')
plt.title('Lotto Winning Numbers Actual vs Prediction')
plt.legend()

plt.show()

# Predict future dates
last_date = df['DrawDate'].max()
future_dates = []
for i in range(1, 6):
    if i == 1:
        last_date += pd.Timedelta(days=3)
    else:
        last_date += pd.Timedelta(days=2)
    future_dates.append(last_date)

future_dates_df = pd.DataFrame({'DrawDate': future_dates})
future_dates_df['DrawDate'] = future_dates_df['DrawDate'].map(pd.Timestamp.toordinal)

# Make predictions for future dates
predicted_numbers_list = []
for date in future_dates_df['DrawDate']:
    future_date = pd.DataFrame({'DrawDate': [date]})
    predicted_numbers = best_model.predict(future_date)
    predicted_numbers = np.clip(predicted_numbers, 1, 42)
    predicted_numbers = np.round(predicted_numbers).astype(int)
    predicted_numbers = make_unique(predicted_numbers[0])
    predicted_numbers_list.append(predicted_numbers)

# Convert ordinal dates back to datetime for display
future_dates_df['DrawDate'] = future_dates_df['DrawDate'].map(pd.Timestamp.fromordinal)

for date, numbers in zip(future_dates_df['DrawDate'], predicted_numbers_list):
    print(f"Predicted winning numbers for {date.date()}: {numbers}")
