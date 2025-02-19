import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt

# Load Data
try:
    df = pd.read_csv('Data//house_prices.csv')
except FileNotFoundError:
    print('File not found!')
    exit()

# Fill missing values for numeric columns with mean
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill missing values for categorical columns with 'Unknown'
categorical_cols = df.select_dtypes(include=[object]).columns
df[categorical_cols] = df[categorical_cols].fillna('Unknown')

# One-Hot Encode Categorical Data
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_features = encoder.fit_transform(df[['Location']])
encoded_feature_names = encoder.get_feature_names_out(['Location'])

# Convert encoded features into a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)

# Concatenate with the original DataFrame
df = pd.concat([df.drop("Location", axis=1), encoded_df], axis=1)

# Save encoder for later use
joblib.dump(encoder, "Model//onehot-encoder.pkl")

# Split data into train and test
X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)

# Scale Numerical Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for future use
joblib.dump(scaler, "Model//scaled-data.pkl")

# Train Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save trained model
joblib.dump(model, "Model//trained-model.pkl")
print("Model saved successfully.")

# Evaluate Model
y_pred = model.predict(X_test_scaled)
print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(f'R² Score: {r2_score(y_test, y_pred)}')

# Prediction on New Data (without manually encoding Location)
def predict_price(new_data):
    # Load the saved encoder & scaler
    loaded_encoder = joblib.load("Model//onehot-encoder.pkl")
    loaded_scaler = joblib.load("Model//scaled-data.pkl")
    loaded_model = joblib.load("Model//trained-model.pkl")

    # Convert new data to DataFrame
    new_df = pd.DataFrame([new_data])

    # Apply one-hot encoding to match training data
    encoded_new_data = loaded_encoder.transform(new_df[['Location']])
    encoded_new_df = pd.DataFrame(encoded_new_data, columns=loaded_encoder.get_feature_names_out(['Location']))

    # Merge encoded data with other features
    new_df = pd.concat([new_df.drop("Location", axis=1), encoded_new_df], axis=1)

    # Scale the features
    new_df_scaled = loaded_scaler.transform(new_df)

    # Predict price
    prediction = loaded_model.predict(new_df_scaled)
    return prediction[0]

# Example: Predicting price for a new house
new_house = {
    "Area (sq ft)": 2500,
    "Bedrooms": 4,
    "Location": "City A"  # No need to manually encode
}

predicted_price = predict_price(new_house)
print(f'Predicted Price for New House: ${predicted_price:,.2f}')

# Plot Actual vs. Predicted Prices
# Sort values by index to maintain time-based order (if applicable)
y_test_sorted = np.array(y_test).flatten()
y_pred_sorted = np.array(y_pred).flatten()
indices = np.arange(len(y_test_sorted))

# Plot actual prices
plt.figure(figsize=(10, 6))
plt.plot(indices, y_test_sorted, label="Actual Prices", color='blue', marker='o', linestyle='dashed')

# Plot predicted prices
plt.plot(indices, y_pred_sorted, label="Predicted Prices", color='red', marker='s', linestyle='dashed')

# Display predicted price values on the graph
for i in range(len(y_pred_sorted)):
    plt.text(indices[i], y_pred_sorted[i], f'{y_pred_sorted[i]:.2f}', fontsize=9, ha='right', color='red')

# Labels and title
plt.xlabel("Data Point Index")
plt.ylabel("House Price")
plt.title("Actual vs. Predicted House Prices")
plt.legend()
plt.grid()
plt.show()


plt.plot(indices, y_test_sorted, label="Actual Prices", color='blue', marker='o', linestyle='dashed')

plt.plot(indices, y_pred_sorted, label="Predicted Prices", color='red', marker='s', linestyle='dashed')

indices = np.arange(len(y_test_sorted))