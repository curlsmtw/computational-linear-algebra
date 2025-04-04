import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Dataset
df = pd.read_csv(r"C:\Users\abbie\OneDrive - CCSU\Documents\CCSU\2025-spring\computational-linear-algebra\projects\project-two\saved-saary-dataset.csv", usecols=["YearsExperience", "Salary"])

df.rename(columns={"YearsExperience": "x", "Salary": "y"}, inplace=True)
print("First few rows of the dataset:")
print(df.head())

#  Extract Variables
x = df['x'].to_numpy().reshape(-1, 1)
y = df['y'].to_numpy().reshape(-1, 1)

#  Construct the Design Matrix 
A = np.hstack([np.ones((x.shape[0], 1)), x])
print("Design Matrix A:")
print(A)

# QR Decomposition for Least Squares
Q, R = np.linalg.qr(A)

# Solve for β in R β = Q^T y
beta = np.linalg.solve(R, Q.T @ y)

# Extract the intercept (b) and slope (m).
b = beta[0, 0]  # intercept
m = beta[1, 0]  # slope
print(f"Intercept (b): {b}")
print(f"Slope (m): {m}")

# Plotting the Data and Fitted Line 
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Data Points')

# Generate x values for plotting the regression line.
x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_line = m * x_line + b

plt.plot(x_line, y_line, color='red', label='Regression Line')
plt.title('Linear Regression via QR Decomposition')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()