
import numpy as np
# 1. Write Python code to create a (4 x 8) matrix of integers from 1 to 32. 
# Extract the element corresponding to your birth month. Extract the element
# corresponding to your birth date. Print the output.

numbers = np.arange(1,33)
matrix_one = numbers.reshape(4, 8)
print("Problem One:\nBirthday Matrix:\n",matrix_one)

month = 1
day = 30

for i in range(matrix_one.shape[0]): #.shape[0] is for number of rows
    for j in range(matrix_one.shape[1]): #.shape[1] is for number of cols
        if matrix_one[i,j] == month:
            print("Birth month:", matrix_one[i,j])
        elif matrix_one[i,j] == day:
            print("Birth day:", matrix_one[i,j])




# 2. Write Python code to create a (10 x 10) matrix C of integers from 1 to 100.
# (a) Extract the submatrix C1 comprising the first five rows and five columns.
# (b) Cut original matrix in 4 blocks such that: C1 is submatrix comprising
# the first five rows and five columns; C2 submatrix comprising the first five
# rows and last five columns; C3 is submatrix comprising the last five rows
# and first five columns. C4 is submatrix comprising the last five rows and five columns.
# Rearrange the blocks to get a new Matrix =

# | C4 C3 |
# | C2 C1 |

matrix_two = np.arange(1,101).reshape(10,10)
print("\nProblem two:\nOriginal Matrix:\n",matrix_two)

c1 = matrix_two[:5, :5]
c2 = matrix_two[:5, 5:]
c3 = matrix_two[5:, :5]
c4 = matrix_two[5:, 5:]
print("First 5 rows and cols:\n", c1)
print("First 5 rows and last 5 cols:\n", c2)
print("Last 5 rows and first 5 cols:\n", c3)
print("Last 5 rows and cols:\n", c4)

final_matrix = np.block([[c4, c3], [c2, c1]])
print ("\n",final_matrix)


# 3. Create two (4 x 4) matrices: a matrix of all ones and a diagonal matrix
# where diagonal elements are 1, 4, 9, and 16. Perform premultiplying and
# posmultiplying of the matrix of ones by the diagonal matrix. Make a conclusion about the result.
ones_matrix = np.ones((4,4))
diag_values = [1, 4, 6, 16]
matrix_three = np.diag(diag_values)
print("\nProblem three:\nThis is the 4 by 4 matrix of ones:\n", ones_matrix, "\n This is the 4 by 4 matrix with the specific diagonal values: \n",matrix_three)
print("Pre:\n", matrix_three@ones_matrix)
print("Post:\n", ones_matrix@matrix_three)

#Conclusion: The values of both the product of premutiplication and postmutiplication 
# shows how the order of the matrices do matter.The values in the premutiplied rows 
# are the same as the postmutiplied columns.

