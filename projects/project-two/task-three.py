import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load the Dataset 
df = pd.read_csv(r"C:\Users\abbie\OneDrive - CCSU\Documents\CCSU\2025-spring\computational-linear-algebra\projects\project-two\saved-saary-dataset.csv",
                 usecols=["YearsExperience", "Salary"])
df.rename(columns={"YearsExperience": "x", "Salary": "y"}, inplace=True)

print("First few rows of the dataset:")
print(df.head())

# Extract Variables 
X = df['x'].values.reshape(-1, 1)
y = df['y'].values.reshape(-1, 1)


# Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the Linear Regression Model to the Training Data
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the Model on the Test Data
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R-squared (R2) score: {r2}")
print(f"Mean Squared Error (MSE): {mse}")

# Plot the Test Data Points and the Regression Line
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Test Data')

# For a smooth regression line, generate a sequence of x values across the test range.
x_line = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)

plt.plot(x_line, y_line, color='red', label='Regression Line')
plt.title('Linear Regression on Test Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()