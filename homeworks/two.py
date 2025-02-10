import numpy as np
import sympy as smp
import scipy.linalg as lg
# Problem 1: Write a Python program to find the Reduced Row Echelon Form (RREF)
# of a given matrix A =1 1 4
#                   −0.5 1 2

matrix_one = np.array([[1, 1, 4], [-1/2,1,2]])
print("Original Matrix", matrix_one)

sym_matrix_one = smp.Matrix(matrix_one)
print("sympy one:\n", sym_matrix_one)

rref_matrix_one = sym_matrix_one.rref()[0]
print("rref(one):\n",rref_matrix_one)


# Problem 2: Write a Python program to find the Reduced Row Echelon Form (RREF)
# of a given matrix
# (a) A =1 2 3
#        4 5 6
#        7 8 9
# (b) B = 2 2 −1 2
#         1 2 1 4
#        −1 −1 2 1

matrix_twoa = np.array([[1,2,3], [4,5,6], [7,8,9]])
print("Original 2A Matrix", matrix_twoa)

sym_matrix_twoa = smp.Matrix(matrix_twoa)
print("sympy 2A:\n", sym_matrix_twoa)

rref_matrix_twoa = sym_matrix_twoa.rref()[0]
print("rref(2A):\n",rref_matrix_twoa)


matrix_twob = np.array([[2,2,-1,2], [1,2,1,4], [-1,-1,2,1]])
print("Original 2B Matrix:", matrix_twob)

sym_matrix_twob = smp.Matrix(matrix_twob)
print("sympy 2b:\n", sym_matrix_twob)

rref_matrix_twob = sym_matrix_twob.rref()[0]
print("rref(2B):\n",rref_matrix_twob)

# Problem 3: Use RREF to solve the linear system A~x = ~b.
# A = 2 2 −1
#     1 2 1
#    −1 −1 2

# b = 2
#     4
#     1



matrix_three_a = smp.Matrix([[2,2,-1], [1,2,1], [-1,-1,2]])
matrix_three_b = smp.Matrix([[2], [4], [1]])

aug_matrix = matrix_three_a.row_join(matrix_three_b)
print("3A Augmented Matrix:\n", aug_matrix)
rref_matrix_three = aug_matrix.rref()[0]
print("RREF of the augmented matrix:", rref_matrix_three)
solution = smp.linsolve((matrix_three_a, matrix_three_b))
print("Soulution:", solution)



# Problem 4: 4. Use an LU -factorization of the coefficient matrix to solve the linear system
# A~x = ~b.
# A = 2 2 −1
#     1 2 1
#    −1 −1 2
# ~b = 2
#      4
#      1
matrix_four_a = np.array([[2,2,-1], [1,2,1], [-1,-1,2]])
matrix_four_b = np.array([[2], [4], [1]])

four_p, four_l, four_u = lg.lu(matrix_four_a)
print("L=\n", four_l, "\nU=\n", four_u, "\nP=\n", four_p)
four_y = lg.solve(four_l, matrix_four_b)
four_x = lg.solve(four_u, four_y)
print("x=\n", four_x)
# Problem 5: 5. Use an LU -factorization of the coefficient matrix to solve the linear system
# A~x = ~b.
# A = 4 −1 −1
#    −1 4 −1
#    −1 −1 4
# ~b = 2
#      2
#      2

matrix_five_a = np.array([[4,-1,-1], [-1,4,-1], [-1,-1,4]])
matrix_five_b = np.array([[2], [2], [2]])

five_p, five_l, five_u = lg.lu(matrix_five_a)
print("L=\n", five_l, "\nU=\n", five_u, "\nP=\n", five_p)
five_y = lg.solve(five_l, matrix_five_b)
five_x = lg.solve(five_u, five_y)
print("x=\n", five_x)

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