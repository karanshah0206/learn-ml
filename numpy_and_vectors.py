import numpy as np
import time

# Allocating memory and filling arrays with values
print("Exploring memory allocation and filling arrays with values")
a = np.zeros(4)
print(f"np.zeros(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,))
print(f"np.zeros((4,)): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.arange(4)
print(f"np.arange(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4)
print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4)
print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5, 4, 3, 2])
print(f"np.array([5, 4, 3, 2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5., 4, 3, 2])
print(f"np.array([5., 4, 3, 2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
print()

# Vector indexing operations in one dimension
print("Exploring vector indexing operations in one dimension using np.arange(10)")
a = np.arange(10)
print(f"a[2] = {a[2]}")
print(f"a[-1] = {a[-1]}")
print(f"a[2].shape = {a[2].shape} The empty shape indicates a scalar value")
try:
    print(f"a[10] = {a[10]}")
except IndexError as e:
    print("Trying a[10], the error message you'll see is:", e)
print()

# Vector slicing operations in one dimension
print("Exploring vector slicing operations in one dimension using np.arange(10)")
a = np.arange(10)
print(f"a[2:7:1] = {a[2:7:1]} (start:stop:step)")
print(f"a[2:7:2] = {a[2:7:2]}")
print(f"a[3:] = {a[3:]}")
print(f"a[:3] = {a[:3]}")
print(f"a[4:-2] = {a[4:-2]}")
print()

# Single vector operations
print("Exploring single vector operations using np.array([1, 2, 3, 4])")
a = np.array([1, 2, 3, 4])
print(f"-a = {-a}")
print(f"np.sum(a) = {np.sum(a)}")
print(f"np.mean(a) = {np.mean(a)}")
print(f"a**2 = {a**2}")
print(f"5 * a = {5 * a}")
print()

# Vector-Vector element-wise operations
print("Exploring vector-vector element-wise operations")
a = np.array([ 1, 2, 3, 4])
b = np.array([-1, -2, 3, 4])
c = np.array([1, 2])
print(f"a = {a}, b = {b}, c = {c}")
print(f"a + b = {a + b}")
try:
    print(f"a + c = {a + c}")
except ValueError as e:
    print("Trying a + c, the error message you'll see is:", e)
print()

# Vector dot product
def my_dot(a: np.ndarray, b: np.ndarray):
    """
    Compute the dot product of two vectors

    Args:
        a (ndarray (n,)): input vector
        b (ndarray (n,)): input vector with same dimension as a

    Returns:
        x (scalar): dot product of vectors a and b
    """
    x=0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x

a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
print(f"Testing dot product for a = {a} and b = {b}")
print(f"np.dot(a, b) = {np.dot(a, b)}")
print(f"my_dot(a, b) = {my_dot(a, b)}")

print(f"Tesitng time taken for dot product calculation on really large vectors generated using np.random.rand(10000000)")
np.random.seed(1)
a = np.random.rand(10000000)
b = np.random.rand(10000000)

start_time = time.time()
c = np.dot(a, b)
stop_time = time.time()
print(f"np.dot(a, b) = {c:.4f} | Duration = {1000*(stop_time-start_time):.4f} ms")

start_time = time.time()
c = my_dot(a, b)
stop_time = time.time()
print(f"my_dot(a, b) = {c:.4f} | Duration = {1000*(stop_time-start_time):.4f} ms")

del(a)
del(b)
