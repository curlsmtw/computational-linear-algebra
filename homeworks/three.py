import numpy as np
import matplotlib.pyplot as plt
import math

#------------------------------------------------------------
# # Consider a linear transformation T : R^2 → R^2 that will 
# # translate points # by (3, -2) and then rotate 45 degrees 
# # about the origin. Come up with a matrix of transformation. 
# # Choose a prototype (can be any but let it have not less
# # than four points) and plot it together with its image.
#------------------------------------------------------------

# Define a prototype shape (a rectangle) with at least four points.
protoype = np.array([[0, 0], [3,0], [3,2], [0,2], [0, 0]])
coords = protoype.transpose()

# Convert to homogeneous coordinates by adding a row of ones.
x = coords [0,:]
y = coords [1,:]
Ones = np.ones((1, 5))
coords = np.vstack((x,y,Ones))

# Plot the original prototype
plt.plot(x,y, 'b', label ='prototype')

# Create a translation matrix to shift points by (3, -2).
T1 = np.array([[1, 0, 3], [0, 1 , -2], [0, 0, 1]])

# Apply the translation.
coords1 = T1@coords
x1 = coords1[0,:]
y1 = coords1[1,:]

# Plot the translated shape 
plt.plot(x1,y1, 'r', label ='translated')

# Create a rotation matrix to rotate by 45° (π/4 radians) about the origin.
th = np.pi/4
T2 = np.array([ [math.cos(th),-math.sin(th),0], [math.sin(th),math.cos(th),0],[0,0,1] ])

# Apply the rotation to the translated coordinates.
coords2=T2@coords1
x2 = coords2[0,:]
y2 = coords2[1,:]

plt.plot(x2,y2,'g',label='rotated')
print('matrix of transformation:'), print(T2@T1)

plt.grid(True)
plt.axis([-6,6,-6,6])
plt.legend()
plt.show()


#------------------------------------------------------------
# # 2. Consider a linear transformation T : R^2 → R^2 that will 
# rotate points 60 degrees # about the point (6,8). Come up with 
# a matrix of transformation. Choose a prototype (can be any but 
# let it have not less than four points) and plot it together 
# with its image.
#------------------------------------------------------------

# Define a new prototype shape (a square) with at least four points.
protoype2 = np.array([[0, 0], [1,0], [1,1], [0,1], [0, 0]])
newcoords = protoype2.transpose()

# Convert to homogeneous coordinates by adding a row of ones.
x3 = newcoords [0,:]
y3 = newcoords [1,:]
Ones2 = np.ones((1, 5))
newcoords = np.vstack((x3,y3,Ones2))

# Plot the original prototype 
plt.plot(x3,y3, 'b', label ='prototype')

# Create a translation matrix to move the rotation center (6,8) to the origin.
T3 = np.array([[1, 0, -6], [0, 1 , -8], [0, 0, 1]])

# Apply the translation.
newcoords1 = T3@newcoords
x4 = newcoords1[0,:]
y4 = newcoords1[1,:]

# Plot the translated shape, now centered at the origin.
plt.plot(x4,y4, 'r', label ='translated to -p')

# Create a rotation matrix to rotate by 60° (π/3 radians) about the origin.
th = np.pi/3
T4 = np.array([ [math.cos(th),-math.sin(th),0], [math.sin(th),math.cos(th),0],[0,0,1] ])

# Apply the rotation.
newcoords2=T4@newcoords1
x5 = newcoords2[0,:]
y5 = newcoords2[1,:]

# Plot the rotated shape 
plt.plot(x5,y5,'g',label='rotated')

# Create a translation matrix to move back from the origin to (6,8).
T5 = np.array([[1, 0, 6], [0, 1 , 8], [0, 0, 1]])

# Apply the translation back.
newcoords3 = T5@newcoords2
x6 = newcoords3[0,:]
y6 = newcoords3[1,:]

# Plot the final image
plt.plot(x6,y6, 'k', label ='image')
print('matrix of transformation:'), print(T5@T4@T3)

plt.grid(True)
plt.axis([-6,6,-6,6])
plt.legend()
plt.show()