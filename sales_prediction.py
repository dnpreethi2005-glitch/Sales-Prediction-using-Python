import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("Advertising.csv")

# Display first 5 rows
print("First 5 rows:")
print(data.head())

# Select input features and output
X = data[['TV', 'radio', 'newspaper']]
y = data['sales']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict sales
y_pred = model.predict(X_test)

# Display predicted values
print("\nPredicted Sales:")
print(y_pred)

# Calculate accuracy
accuracy = r2_score(y_test, y_pred)
print("\nAccuracy (R² Score):", accuracy)

# Plot Actual vs Predicted Sales
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Predict sales for new advertising values
new_data = pd.DataFrame({
    'TV': [230.1],
    'radio': [37.8],
    'newspaper': [69.2]
})

prediction = model.predict(new_data)

print("\nPredicted Sales for New Data:", prediction[0])