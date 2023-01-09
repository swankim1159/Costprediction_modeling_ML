# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load the training data
train_data = pd.read_csv("train.csv")
X = train_data.drop(columns=['target'])
y = train_data['target']

# Train a random forest model
model = RandomForestRegressor()
model.fit(X, y)

# Load the test data
test_data = pd.read_csv("test.csv")
X_test = test_data.drop(columns=['target'])

# Make predictions on the test data
y_pred = model.predict(X_test)

# Add the predictions to the test data
test_data['target'] = y_pred

# Save the updated test data to a new CSV file
test_data.to_csv("test_updated.csv", index=False)
