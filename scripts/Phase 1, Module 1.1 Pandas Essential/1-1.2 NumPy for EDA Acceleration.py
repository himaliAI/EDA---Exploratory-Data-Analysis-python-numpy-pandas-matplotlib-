import numpy as np

a = np.arange(1, 11)

a_squares = a ** 2

a_cube_roots = a ** (1/3)

a_multiply10 = a * 10

# Broadcasting
# sclar with array
arr = np.array([1, 2, 3, 4, 5])
add_ten = arr + 10
arr_double = arr * 2

# row vector and column vector
row = np.array([1, 2, 3]) # shape (3,)
col = np.array([            # shape (3, 1)
    [10],
    [20],
    [30]
])
addition = row + col # shape (3, 3); 

# matrix with vector
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
vector = np.array([10, 20, 30])
add = matrix + vector # shape (2, 3)

# random number generation in NumPy
rng = np.random.default_rng(seed=123)
    # np.random.seed(123)

# uniform random numbers (0-1)
uniform = rng.uniform(size=5) # 1D array of 5 numbers 0-1
uniform2 = np.random.rand(5)
uniform3 = rng.uniform(1, 10, 5) # 1D array of 5 numbers 1-10
uniform4 = rng.uniform(size=(3, 4))

# random integers
integers = rng.integers(1, 101, 10) # 10 random integers 1-100
integers2 = np.random.randint(101, 201, 10)
integers3 = rng.integers(1, 10, size=(3, 3))

# normal distribution
normal = rng.normal(10, 2.5, 10)
normal1 = np.random.normal(0, 1, 5)
normal2 = rng.normal(100, 25, size=(3,4))

