
import numpy as np
# 1. Write Python code to create a (4 x 8) matrix of integers from 1 to 32. 
# Extract the element corresponding to your birth month. Extract the element
# corresponding to your birth date. Print the output.

numbers = np.arange(1,33)
matrix = numbers.reshape(4, 8)
print(matrix)

month = 1
day = 30

for i in range(matrix.shape[0]): #.shape[0] is for number of rows
    for j in range(matrix.shape[1]): #.shape[1] is for number of cols
        if matrix[i,j] == month:
            print("Birth month:", matrix[i,j])
        elif matrix[i,j] == day:
            print("Birth day:", matrix[i,j])




# 2. Write Python code to create a (10 x 10) matrix C of integers from 1 to
# 100.
# (a) Extract the submatrix C1 comprising the first five rows and five
# columns.
# (b) Cut original matrix in 4 blocks such that: C1 is submatrix comprising
# the first five rows and five columns; C2 submatrix comprising the first five
# rows and last five columns; C3 is submatrix comprising the last five rows
# and first five columns. C4 is submatrix comprising the last five rows and
# five columns.
# Rearrange the blocks to get a newM atrix =

# | C4 C3 |
# | C2 C1 |

# 3. Create two (4 x 4) matrices: a matrix of all ones and a diagonal matrix
# where diagonal elements are 1, 4, 9, and 16. Perform premultiplying and
# posmultiplying of the matrix of ones by the diagonal matrix. Make a
# conclusion about the result.