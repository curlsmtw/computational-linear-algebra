import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Consider the following vectors in R^3:
# u1 = [ 1, 3, −2]
# u2 = [ 5, 1, 4 ]
#  y = [ 1, 4, 5]

u1 = np.array([1, 3, -2])
u2 = np.array([5, 1, 4])
y = np.array([1, 4, 5])

# Determines if 2 vectors are orthogonal
def isortho(v1, v2):
    if np.dot(v1,v2) == 0:
        print("The vectors are orthogonal.")
    else:
        print("The vectors are not orthogonal.")


# (a) Find the best approximation of y in the subspace W = Span {u1, u2},
# which is the projection of y onto the subspace spanned by u1 and
# u2. 

# Checks if 2 vectors are orthogonal
isortho(u1, u2)

# Projection
proj_y = (np.dot(y, u1)/np.dot(u1, u1)) * u1 + (np.dot(y, u2)/np.dot(u2, u2)) * u2
orth_component = y - proj_y
print("\nProjection of y onto W:")
print(proj_y)
print("\nOrthogonal component (y - projection):")
print(orth_component)


# (b) Visualize the subspace spanned by the orthogonalized vectors and the
# projection of y onto this subspace in a 3D plot. Include the following
# in your plot:
# • The vectors u1 and u2,
# • The vector y,
# • The projection of y onto the subspace spanned by u1 and u2,
# • The orthogonal complement of y relative to the subspace.

# New figure for 3D plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
origin = np.array([0, 0, 0])

# Plot the vectors (arrows starting from the origin)
ax.quiver(*origin, *u1, color='red', label='u1', arrow_length_ratio=0.1)
ax.quiver(*origin, *u2, color='green', label='u2', arrow_length_ratio=0.1)
ax.quiver(*origin, *y,  color='blue', label='y', arrow_length_ratio=0.1)
ax.quiver(*origin, *proj_y, color='magenta', label='Projection of y', arrow_length_ratio=0.1)

# Plot the orthogonal component starting from the tip of the projection
ax.quiver(*proj_y, *(y - proj_y), color='black', label='y - proj_y', arrow_length_ratio=0.1)

# Create a grid to plot the subspace (plane) spanned by u1 and u2.
s_vals = np.linspace(-5, 5, 10)
t_vals = np.linspace(-5, 5, 10)
S, T = np.meshgrid(s_vals, t_vals)

# For each (s, t), compute p = s*u1 + t*u2.
plane_points = np.zeros((S.shape[0], S.shape[1], 3))
for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        plane_points[i, j, :] = S[i, j] * u1 + T[i, j] * u2

X = plane_points[:, :, 0]
Y_plane = plane_points[:, :, 1]
Z = plane_points[:, :, 2]

ax.plot_surface(X, Y_plane, Z, alpha=0.3, color='cyan')

# Set plot labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Projection of y onto Subspace W = Span{u1, u2}')
ax.legend()

# Displays the plot
plt.show()

# 2. For the 4 × 3 matrix A = [ 3 −5 1,
#                               1 1 1,
#                               −1 5 −2,
#                               3 −7 8],
# find an orthonormal basis for Col(A).


A = np.array([[3,-5,1], [1,1,1], [-1,5,-2], [3,-7,8]])

# Extract the columns as vectors.
a1 = A[:, 0]
a2 = A[:, 1]
a3 = A[:, 2]

def normalize(v):
    """Return the normalized vector of v."""
    return v / np.linalg.norm(v)

# Apply the Gram–Schmidt process:

# Normalize a1.
v1 = normalize(a1)

# Orthogonalize a2 against v1.
proj_v1_a2 = np.dot(a2, v1) * v1
w2 = a2 - proj_v1_a2
v2 = normalize(w2)

# Orthogonalize a3 against v1 and v2.
proj_v1_a3 = np.dot(a3, v1) * v1
proj_v2_a3 = np.dot(a3, v2) * v2
w3 = a3 - proj_v1_a3 - proj_v2_a3
v3 = normalize(w3)

print("\nOrthonormal basis for Col(A):")
print("v1 =", v1)
print("v2 =", v2)
print("v3 =", v3)