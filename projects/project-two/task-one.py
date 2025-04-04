import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Dataset
df = pd.read_csv(r"C:\Users\abbie\OneDrive - CCSU\Documents\CCSU\2025-spring\computational-linear-algebra\projects\project-two\saved-saary-dataset.csv", usecols=["YearsExperience", "Salary"])
df.rename(columns={"YearsExperience": "x", "Salary": "y"}, inplace=True)

print("First few rows of the dataset:")
print(df.head())


# Extract predictor variable x and target variable y.
x = df['x'].to_numpy()          # Predictor variable
y = df['y'].to_numpy()          # Target variable 

# Reshape x and y into column vectors.
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# For linear regression with an intercept, the design matrix X includes a column of ones.
ones = np.full((len(x), 1), 1)
X = np.append(ones, x, axis=1)
print("Design Matrix X:")
print(X)

# Linear Algebra Least Squares Calculation 
beta = np.linalg.inv(X.T @ X) @ (X.T @ y)

# Extract the slope (m) and intercept (b) from beta.
b = beta[0]
m = beta[1]
print(f"Slope (m): {m[0]}")
print(f"Intercept (b): {b[0]}")

# Plotting the Data and Fitted Line 
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Data Points')
# Create a range of x values for plotting the regression line
x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_line = m * x_line + b
plt.plot(x_line, y_line, color='red', label='Fitted Line')
plt.title('Linear Regression using Least Squares Method')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
