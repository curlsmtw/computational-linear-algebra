import numpy as np
import matplotlib.pyplot as plt

# Problem 1: Discrete Price Model

# Data: time (t) and price (p)
time_price = np.array([1, 3, 5, 7, 9, 11, 13])
price = np.array([10000, 9000, 8200, 7400, 6700, 6200, 6000])

# (a) Construct the least squares system A*x = d
# Design matrix A has a column of ones and a column of time values.
A = np.column_stack((np.ones(time_price.shape), time_price))
d = price

print("Problem 1: Discrete Price Model")
print("Design Matrix A:")
print(A)
print("Data vector d (prices):")
print(d)

# (b) Solve for the least squares solution x = [β0, β1]
# Using numpy's lstsq function.
x, residuals, rank, s = np.linalg.lstsq(A, d, rcond=None)
beta0, beta1 = x
print("\nLeast Squares Solution:")
print("β0 =", beta0, "β1 =", beta1)
print("Residual sum of squares:", residuals)

# (c) Predict the price at time t = 15
t_pred = 15
price_pred = beta0 + beta1 * t_pred
print("\nPredicted price at t = 15:", price_pred)

# Plot the data and best-fit line
plt.figure(figsize=(8, 6))
plt.scatter(time_price, price, color='blue', label='Data Points')
t_line = np.linspace(time_price.min() - 1, t_pred + 1, 100)
price_line = beta0 + beta1 * t_line
plt.plot(t_line, price_line, color='red', label='Best Fit Line')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Least Squares Fit for Discrete Price Model')
plt.legend()
plt.grid(True)
plt.show()


# Problem 2: Airplane Takeoff Performance

# Data: Time (in seconds) and horizontal positions (in feet)
t_air = np.arange(0, 13)  # t = 0, 1, 2, ... 12 seconds
position = np.array([0, 8.8, 29.9, 62.0, 104.7, 159.1, 222.0, 294.5,
                     380.4, 471.1, 571.7, 686.8, 809.2])

print("\nProblem 2: Airplane Takeoff Performance")
print("Time data:", t_air)
print("Position data:", position)

# (a) Fit a cubic polynomial: y = β0 + β1*t + β2*t^2 + β3*t^3
# Construct the design matrix for a cubic model.
A_cubic = np.column_stack((np.ones(t_air.shape), t_air, t_air**2, t_air**3))
# Solve the least squares problem.
beta_cubic, residuals, rank, s = np.linalg.lstsq(A_cubic, position, rcond=None)
print("Cubic Polynomial Coefficients (β0, β1, β2, β3):")
print(beta_cubic)

# (b) Compute the velocity at t = 4.5 seconds.
# For the cubic: y = β0 + β1*t + β2*t^2 + β3*t^3,
# the velocity is the derivative: v(t) = β1 + 2β2*t + 3β3*t^2.
t_velocity = 4.5
velocity = beta_cubic[1] + 2 * beta_cubic[2] * t_velocity + 3 * beta_cubic[3] * t_velocity**2
print("Velocity at t = 4.5 seconds:", velocity)

# Plot the airplane position data and the fitted cubic curve
t_dense = np.linspace(t_air.min(), t_air.max(), 200)
A_dense = np.column_stack((np.ones(t_dense.shape), t_dense, t_dense**2, t_dense**3))
position_fit = A_dense @ beta_cubic

plt.figure(figsize=(8, 6))
plt.scatter(t_air, position, color='green', label='Data Points')
plt.plot(t_dense, position_fit, color='orange', label='Cubic Fit')
plt.xlabel('Time (seconds)')
plt.ylabel('Position (feet)')
plt.title('Least Squares Cubic Fit for Airplane Takeoff Performance')
plt.legend()
plt.grid(True)
plt.show()
