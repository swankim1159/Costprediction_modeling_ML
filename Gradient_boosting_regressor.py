# Import necessary libraries
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Load the training data
train_data = pd.read_csv("train.csv")
X = train_data.drop(columns=['target'])
y = train_data['target']

# Train a gradient boosting model
model = GradientBoostingRegressor()
model.fit(X, y)

# Load the test data
test_data = pd.read_csv("test.csv")
X_test = test_data.drop(columns=['target'])

# Make predictions on the test data
y_pred = model.predict(X_test)

# Add the predictions to the test data
test_data['target'] = y_pred

# Save the updated test data to a new CSV file
test_data.to_csv("test_updated_gbr.csv", index=False)

# Compute the mean absolute error
mae = mean_absolute_error(test_data['target'], y)
print("MAE:", mae)

# Check if the MAE is less than 0.0015
if mae < 0.0015:
    print("MAE passes the requirement")
else:
    print("MAE does not pass the requirement")