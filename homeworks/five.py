import numpy as np

# 1-a: Compute the QR decomposition of the matrix 
# A = [2 −1 1
# 	   1 3 2
#      1 −1 2]

# Define the matrix A
A1 = np.array([[2,-1,1],
              [1, 3, 2],
              [1, -1, 2]])

# Perform QR decomposition
Q1, R1 = np.linalg.qr(A1)
print("Solution to 1(a): ")
print("\n", np.round(Q1,2), "\n")
print(np.round(R1,2), "\n")


# 1-b: Use the QR factorization to compute det(A)
# Compute the determinant from QR decomposition (det(A) = product of diagonal elements of R)
det_from_qr1 = np.prod(np.diagonal(R1))

# Display the results
print(f"Solution to 1(b):\n {np.abs(det_from_qr1)} \n")


# 1-c: Use QR factorizaztion to solve teh system Ax = b for:
# b = [1
# 	 4
# 	 3]
b1 = np.array([[1], [4], [3]])
y1 = np.dot(Q1.T, b1)

x1 = np.linalg.solve(R1, y1)

print("Solution to 1(c):\n ", x1)


# 2-a: Compute the QR decomposition of the matrix Ax = b, where 
# A = [4 −1 −1
# 	−1 4 −1
# 	−1 −1 4]
A2 = np.array([[4, -1, -1], 
              [-1, 4, -1],
              [-1, -1, 4]])

Q2, R2 = np.linalg.qr(A2)

print("Solution to 2(a): ")
print(np.round(Q2, 2),"\n")
print(np.round(R2, 2),"\n")

# 2-b: Use QR factorization to compute det(A);
det_from_qr2 = np.prod(np.diagonal(R2))
print(f"Solution to 2(b):\n {np.abs(det_from_qr2)} \n")


# 2-c: Use QR factorization to compute A^−1
Ai1=np.linalg.inv(R2)@Q2.T
# solves RA^{-1}=Q^{T} to find A^{-1}
Ai2=np.linalg.solve(R2,Q2.T)
print("Solution to 2(c): ")
print(Ai2, "\n")


# 2-d: Use QR factorization to solve the linear system and 
# b = [2
# 	 2
# 	 2]
b2 = np.array([[2], [2], [2]])
y2 = np.dot(Q2.T, b2)

x2 = np.linalg.solve(R2, y2)

print("Solution to 2(d):\n ", x2)