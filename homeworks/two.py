import numpy as mp
import sympy as smp
# Problem 1: Write a Python program to find the Reduced Row Echelon Form (RREF)
# of a given matrix A =1 1 4
#                   −0.5 1 2

# Problem 2: Write a Python program to find the Reduced Row Echelon Form (RREF)
# of a given matrix
# (a) A =1 2 3
#        4 5 6
#        7 8 9
# (b) B = 2 2 −1 2
#         1 2 1 4
#        −1 −1 2 1

# Problem 3: Use RREF to solve the linear system A~x = ~b.
# A = 2 2 −1
#     1 2 1
#    −1 −1 2

# b = 2
#     4
#     1

# Problem 4: 4. Use an LU -factorization of the coefficient matrix to solve the linear system
# A~x = ~b.
# A = 2 2 −1
#     1 2 1
#    −1 −1 2
# ~b = 2
#      4
#      1

# Problem 5: 5. Use an LU -factorization of the coefficient matrix to solve the linear system
# A~x = ~b.
# A = 4 −1 −1
#    −1 4 −1
#    −1 −1 4
# ~b = 2
#      2
#      2

# Problem 6: One application of LU is computing the determinant. 
# We know that the determinant of a triangular matrix is the product 
# of the diagonals. We also know that the determinant of the product
# matrix equals the product of the determinants. Putting these two
# facts together, we can compute the determinant of a matrix A = P
# TLU as the product of the diagonals of L times the product of the
# diagonals of U times det(P). On the other hand, because the diagonals
# of L are all ones, then the determinant of A is simply the product of 
# the diagonals of U multiplied by the determinant of permutation  matrix.
# Try it in Python and compare the result to the result of np.linalg.det(A)