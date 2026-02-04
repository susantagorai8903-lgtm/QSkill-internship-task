import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np


train_data = pd.read_csv('c:/Users/user/OneDrive/Desktop/Qskill Internship/task2/house_price_train.csv')
test_data = pd.read_csv('c:/Users/user/OneDrive/Desktop/Qskill Internship/task2/house_price_test.csv')

numeric_columns = train_data.select_dtypes(include=['number']).columns
categorical_columns = train_data.select_dtypes(include=['object']).columns

train_data[numeric_columns] = train_data[numeric_columns].fillna(train_data[numeric_columns].mean())
test_data[numeric_columns] = test_data[numeric_columns].fillna(test_data[numeric_columns].mean())

train_data[categorical_columns] = train_data[categorical_columns].fillna(train_data[categorical_columns].mode().iloc[0])
test_data[categorical_columns] = test_data[categorical_columns].fillna(test_data[categorical_columns].mode().iloc[0])


encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_train_location = encoder.fit_transform(train_data[['Location']])
encoded_test_location = encoder.transform(test_data[['Location']])

encoded_train_df = pd.DataFrame(encoded_train_location, columns=encoder.get_feature_names_out(['Location']))
encoded_test_df = pd.DataFrame(encoded_test_location, columns=encoder.get_feature_names_out(['Location']))

train_data = pd.concat([train_data, encoded_train_df], axis=1).drop('Location', axis=1)
test_data = pd.concat([test_data, encoded_test_df], axis=1).drop('Location', axis=1)

scaler = StandardScaler()
numerical_features = ['Number_of_Rooms', 'Size_sqft']
train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])
test_data[numerical_features] = scaler.transform(test_data[numerical_features])

test_data = test_data.drop(columns=['House_Price'], errors='ignore')

X = train_data.drop(['House_Price'], axis=1)
y = train_data['House_Price']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

val_predictions = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f"Validation RMSE: {rmse}")

test_predictions = model.predict(test_data)

# Save predictions to CSV
output = pd.DataFrame({'House_Price': test_data.index + 1, 'Predicted_Price': test_predictions})
output.to_csv('house_price_predictions.csv', index=False)
print("Predictions saved to house_price_predictions.csv")

