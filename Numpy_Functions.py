import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
Numpy=Image.open("Numpylgo.jpg")
st.image(Numpy,use_column_width=True)

# Set the title
st.title("A Deep Dive into NumPy: Functions Explained with Examples")

# Dropdown menu for NumPy functions
functions_list = [
    "np.flatten", "np.ravel", "np.where", "np.concatenate", "np.vstack", "np.hstack",
    "np.sum", "np.prod", "np.pow", "np.cumsum", "np.cumprod", "np.nansum", "np.nanprod",
    "np.shape", "np.ndim", "np.seed", "np.floor", "np.ceil", "np.max", "np.min", "np.var",
    "np.mean", "np.median", "np.percentile", "np.argmin", "np.argmax", "np.newaxis",
    "np.reshape", "Array Broadcasting", "Logarithmic Functions", "np.choice", "np.imshow",
    "np.imread", "np.quantile", "np.any", "np.all", "np.eye", "np.diag", "np.full", "np.dot",
    "np.matrix", "np.linalg.inv", "np.linalg.det", "np.transpose", "Trigonometric Functions","np.apply_along_axis",
    "np.char.upper","np.char.lower","np.char.title","np.char.capitalize","np.char.split","np.char.strip",
    "np.char.join","np.exp","np.std"
]

selected_function = st.selectbox("Select a NumPy function to learn about:", functions_list)

# Dictionary with function descriptions, examples, and output
examples = {
    "np.array": {
        "syntax": "np.array(object)",
        "description": "Creates an array.",
        "example_code": "np.array([1, 2, 3, 4, 5])",
        "example": lambda: np.array([1, 2, 3, 4, 5])
    },
    "np.arange": {
        "syntax": "np.arange(start, stop, step)",
        "description": "Returns evenly spaced values within a given interval.",
        "example_code": "np.arange(0, 10, 2)",
        "example": lambda: np.arange(0, 10, 2)
    },
    "np.linspace": {
        "syntax": "np.linspace(start, stop, num)",
        "description": "Returns evenly spaced numbers over a specified interval.",
        "example_code": "np.linspace(0, 1, 5)",
        "example": lambda: np.linspace(0, 1, 5)
    },
    "np.zeros": {
        "syntax": "np.zeros(shape)",
        "description": "Returns a new array of given shape and type, filled with zeros.",
        "example_code": "np.zeros((2, 2))",
        "example": lambda: np.zeros((2, 2))
    },
    "np.ones": {
        "syntax": "np.ones(shape)",
        "description": "Returns a new array of given shape and type, filled with ones.",
        "example_code": "np.ones((2, 3))",
        "example": lambda: np.ones((2, 3))
    },
    "np.eye": {
        "syntax": "np.eye(N)",
        "description": "Returns a 2-D array with ones on the diagonal and zeros elsewhere.",
        "example_code": "np.eye(3)",
        "example": lambda: np.eye(3)
    },
    "np.random.rand": {
        "syntax": "np.random.rand(d0, d1, ..., dn)",
        "description": "Creates an array of the given shape and populates it with random samples from a uniform distribution over [0, 1).",
        "example_code": "np.random.rand(2, 3)",
        "example": lambda: np.random.rand(2, 3)
    },
    "np.dot": {
        "syntax": "np.dot(a, b)",
        "description": "Dot product of two arrays.",
        "example_code": "np.dot(np.array([1, 2]), np.array([3, 4]))",
        "example": lambda: np.dot(np.array([1, 2]), np.array([3, 4]))
    },
    "np.sum": {
        "syntax": "np.sum(a, axis=None)",
        "description": "Sum of array elements over a given axis.",
        "example_code": "np.sum([1, 2, 3, 4, 5])",
        "example": lambda: np.sum([1, 2, 3, 4, 5])
    },
    "np.prod": {
        "syntax": "np.prod(a, axis=None)",
        "description": "Product of array elements over a given axis.",
        "example_code": "np.prod([1, 2, 3, 4, 5])",
        "example": lambda: np.prod([1, 2, 3, 4, 5])
    },
    "np.pow": {
        "syntax": "np.power(a, b)",
        "description": "First array elements raised to powers from second array, element-wise.",
        "example_code": "np.power([1, 2, 3], 2)",
        "example": lambda: np.power([1, 2, 3], 2)
    },
    "np.cumsum": {
        "syntax": "np.cumsum(a, axis=None)",
        "description": "Cumulative sum of the elements along a given axis.",
        "example_code": "np.cumsum([1, 2, 3, 4, 5])",
        "example": lambda: np.cumsum([1, 2, 3, 4, 5])
    },
    "np.cumprod": {
        "syntax": "np.cumprod(a, axis=None)",
        "description": "Cumulative product of the elements along a given axis.",
        "example_code": "np.cumprod([1, 2, 3, 4, 5])",
        "example": lambda: np.cumprod([1, 2, 3, 4, 5])
    },
    "np.nansum": {
        "syntax": "np.nansum(a, axis=None)",
        "description": "Sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.",
        "example_code": "np.nansum([1, 2, np.nan, 4, 5])",
        "example": lambda: np.nansum([1, 2, np.nan, 4, 5])
    },
    "np.nanprod": {
        "syntax": "np.nanprod(a, axis=None)",
        "description": "Product of array elements over a given axis treating Not a Numbers (NaNs) as ones.",
        "example_code": "np.nanprod([1, 2, np.nan, 4, 5])",
        "example": lambda: np.nanprod([1, 2, np.nan, 4, 5])
    },
    "np.flatten": {
        "syntax": "np.flatten()",
        "description": "Returns a copy of the array collapsed into one dimension.",
        "example_code": "np.array([[1, 2], [3, 4]]).flatten()",
        "example": lambda: np.array([[1, 2], [3, 4]]).flatten()
    },
    "np.ravel": {
        "syntax": "np.ravel(a)",
        "description": "Returns a contiguous flattened array.",
        "example_code": "np.ravel([[1, 2], [3, 4]])",
        "example": lambda: np.ravel([[1, 2], [3, 4]])
    },
    "np.where": {
        "syntax": "np.where(condition, [x, y])",
        "description": "Return elements chosen from `x` or `y` depending on `condition`.",
        "example_code": "np.where(np.array([1, 2, 3]) > 1, 'yes', 'no')",
        "example": lambda: np.where(np.array([1, 2, 3]) > 1, 'yes', 'no')
    },
    "np.concatenate": {
        "syntax": "np.concatenate((a1, a2, ...), axis=0)",
        "description": "Join a sequence of arrays along an existing axis.",
        "example_code": "np.concatenate(([1, 2], [3, 4], [5, 6]))",
        "example": lambda: np.concatenate(([1, 2], [3, 4], [5, 6]))
    },
    "np.vstack": {
        "syntax": "np.vstack(tup)",
        "description": "Stack arrays in sequence vertically (row-wise).",
        "example_code": "np.vstack(([1, 2], [3, 4]))",
        "example": lambda: np.vstack(([1, 2], [3, 4]))
    },
    "np.hstack": {
        "syntax": "np.hstack(tup)",
        "description": "Stack arrays in sequence horizontally (column-wise).",
        "example_code": "np.hstack(([1, 2], [3, 4]))",
        "example": lambda: np.hstack(([1, 2], [3, 4]))
    },
    "np.shape": {
        "syntax": "a.shape",
        "description": "Returns the shape of an array.",
        "example_code": "np.array([[1, 2], [3, 4]]).shape",
        "example": lambda: np.array([[1, 2], [3, 4]]).shape
    },
    "np.ndim": {
        "syntax": "a.ndim",
        "description": "Returns the number of dimensions of an array.",
        "example_code": "np.array([[1, 2], [3, 4]]).ndim",
        "example": lambda: np.array([[1, 2], [3, 4]]).ndim
    },
    "np.seed": {
        "syntax": "np.random.seed(seed)",
        "description": "Sets the seed for random number generation.",
        "example_code": "np.random.seed(42); np.random.rand(2)",
        "example": lambda: np.random.seed(42) or np.random.rand(2)
    },
    "np.floor": {
        "syntax": "np.floor(x)",
        "description": "Return the floor of the input, element-wise.",
        "example_code": "np.floor([1.7, 2.3, 3.9])",
        "example": lambda: np.floor([1.7, 2.3, 3.9])
    },
    "np.ceil": {
        "syntax": "np.ceil(x)",
        "description": "Return the ceiling of the input, element-wise.",
        "example_code": "np.ceil([1.7, 2.3, 3.9])",
        "example": lambda: np.ceil([1.7, 2.3, 3.9])
    },
    "np.max": {
        "syntax": "np.max(a)",
        "description": "Returns the maximum of an array or maximum along an axis.",
        "example_code": "np.max([1, 2, 3, 4, 5])",
        "example": lambda: np.max([1, 2, 3, 4, 5])
    },
    "np.min": {
        "syntax": "np.min(a)",
        "description": "Returns the minimum of an array or minimum along an axis.",
        "example_code": "np.min([1, 2, 3, 4, 5])",
        "example": lambda: np.min([1, 2, 3, 4, 5])
    },
    "np.var": {
        "syntax": "np.var(a)",
        "description": "Returns the variance of an array.",
        "example_code": "np.var([1, 2, 3, 4, 5])",
        "example": lambda: np.var([1, 2, 3, 4, 5])
    },
    "np.mean": {
        "syntax": "np.mean(a)",
        "description": "Returns the mean of an array.",
        "example_code": "np.mean([1, 2, 3, 4, 5])",
        "example": lambda: np.mean([1, 2, 3, 4, 5])
    },
    "np.median": {
        "syntax": "np.median(a)",
        "description": "Returns the median of an array.",
        "example_code": "np.median([1, 2, 3, 4, 5])",
        "example": lambda: np.median([1, 2, 3, 4, 5])
    },
    "np.percentile": {
        "syntax": "np.percentile(a, q)",
        "description": "Compute the q-th percentile of the data along the specified axis.",
        "example_code": "np.percentile([1, 2, 3, 4, 5], 50)",
        "example": lambda: np.percentile([1, 2, 3, 4, 5], 50)
    },
    "np.argmin": {
        "syntax": "np.argmin(a)",
        "description": "Returns the indices of the minimum values along an axis.",
        "example_code": "np.argmin([1, 2, 3, 4, 5])",
        "example": lambda: np.argmin([1, 2, 3, 4, 5])
    },
    "np.argmax": {
        "syntax": "np.argmax(a)",
        "description": "Returns the indices of the maximum values along an axis.",
        "example_code": "np.argmax([1, 2, 3, 4, 5])",
        "example": lambda: np.argmax([1, 2, 3, 4, 5])
    },
    "np.newaxis": {
        "syntax": "np.newaxis",
        "description": "Used to increase the dimension of the existing array by one more dimension, when used once.",
        "example_code": "np.array([1, 2, 3])[:, np.newaxis]",
        "example": lambda: np.array([1, 2, 3])[:, np.newaxis]
    },
    "np.reshape": {
        "syntax": "np.reshape(a, newshape)",
        "description": "Gives a new shape to an array without changing its data.",
        "example_code": "np.reshape(np.array([1, 2, 3, 4]), (2, 2))",
        "example": lambda: np.reshape(np.array([1, 2, 3, 4]), (2, 2))
    },
    "Array Broadcasting": {
        "syntax": "Broadcasting rules",
        "description": "Refers to how NumPy handles element-wise operations with arrays of different shapes.",
        "example_code": "np.array([1, 2, 3]) + np.array([[1], [2], [3]])",
        "example": lambda: np.array([1, 2, 3]) + np.array([[1], [2], [3]])
    },
    "Logarithmic Functions": {
        "syntax": "np.log(a), np.log10(a), np.log2(a)",
        "description": "Computes the natural, base-10, or base-2 logarithm of array elements.",
        "example_code": "np.log([1, np.e, np.e**2]), np.log10([1, 10, 100]), np.log2([1, 2, 4])",
        "example": lambda: (np.log([1, np.e, np.e**2]), np.log10([1, 10, 100]), np.log2([1, 2, 4]))
    },
    "np.choice": {
        "syntax": "np.random.choice(a, size=None, replace=True, p=None)",
        "description": "Generates a random sample from a given 1-D array.",
        "example_code": "np.random.choice([1, 2, 3, 4, 5], size=3)",
        "example": lambda: np.random.choice([1, 2, 3, 4, 5], size=3)
    },
    "np.imshow": {
        "syntax": "plt.imshow(X, cmap=None)",
        "description": "Displays an image represented by an array.",
        "example_code": 'plt.imshow(plt.imread("Inno.png"))',
        "example": lambda: plt.imshow(plt.imread("Inno.png"))
    },
    "np.imread": {
        "syntax": "plt.imread(fname)",
        "description": "Reads an image from a file into an array.",
        "example_code": "plt.imread('image.png')",
        "example": lambda: "Image read function (use your own file for testing)"
    },
    "np.quantile": {
        "syntax": "np.quantile(a, q)",
        "description": "Computes the q-th quantile of the data along the specified axis.",
        "example_code": "np.quantile([1, 2, 3, 4, 5], 0.5)",
        "example": lambda: np.quantile([1, 2, 3, 4, 5], 0.5)
    },
    "np.any": {
        "syntax": "np.any(a)",
        "description": "Tests whether any array element along a given axis evaluates to True.",
        "example_code": "np.any([False, False, True])",
        "example": lambda: np.any([False, False, True])
    },
    "np.all": {
        "syntax": "np.all(a)",
        "description": "Tests whether all array elements along a given axis evaluate to True.",
        "example_code": "np.all([True, True, True])",
        "example": lambda: np.all([True, True, True])
    },
    "np.eye": {
        "syntax": "np.eye(N, M=None, k=0)",
        "description": "Returns a 2-D array with ones on the diagonal and zeros elsewhere.",
        "example_code": "np.eye(3)",
        "example": lambda: np.eye(3)
    },
    "np.diag": {
        "syntax": "np.diag(v, k=0)",
        "description": "Extracts or constructs a diagonal array.",
        "example_code": "np.diag([1, 2, 3])",
        "example": lambda: np.diag([1, 2, 3])
    },
    "np.full": {
        "syntax": "np.full(shape, fill_value)",
        "description": "Returns a new array of given shape and type, filled with fill_value.",
        "example_code": "np.full((2, 3), 7)",
        "example": lambda: np.full((2, 3), 7)
    },
    "np.dot": {
        "syntax": "np.dot(a, b)",
        "description": "Dot product of two arrays.",
        "example_code": "np.dot([1, 2], [3, 4])",
        "example": lambda: np.dot([1, 2], [3, 4])
    },
    "np.matrix": {
        "syntax": "np.matrix(data, dtype=None)",
        "description": "Returns a matrix from an array-like object, or from a string of data.",
        "example_code": "np.matrix([[1, 2], [3, 4]])",
        "example": lambda: np.matrix([[1, 2], [3, 4]])
    },
    "np.linalg.inv": {
        "syntax": "np.linalg.inv(a)",
        "description": "Computes the (multiplicative) inverse of a matrix.",
        "example_code": "np.linalg.inv(np.array([[1, 2], [3, 4]]))",
        "example": lambda: np.linalg.inv(np.array([[1, 2], [3, 4]]))
    },
    "np.linalg.det": {
        "syntax": "np.linalg.det(a)",
        "description": "Computes the determinant of an array.",
        "example_code": "np.linalg.det(np.array([[1, 2], [3, 4]]))",
        "example": lambda: np.linalg.det(np.array([[1, 2], [3, 4]]))
    },
    "np.transpose": {
        "syntax": "np.transpose(a, axes=None)",
        "description": "Permutes the dimensions of an array.",
        "example_code": "np.transpose(np.array([[1, 2], [3, 4]]))",
        "example": lambda: np.transpose(np.array([[1, 2], [3, 4]]))
    },
    "Trigonometric Functions": {
        "syntax": "np.sin(a), np.cos(a), np.tan(a)",
        "description": "Computes the trigonometric sine, cosine, and tangent of array elements.",
        "example_code": "np.sin(np.pi/2), np.cos(0), np.tan(np.pi/4)",
        "example": lambda: (np.sin(np.pi/2), np.cos(0), np.tan(np.pi/4))
    },
    "np.apply_along_axis": {
        "syntax": "np.apply_along_axis(funcid,axis,arr)",
        "description": "np.apply_along_axis() allows you to apply a function along the specified axis of a NumPy array",
        "example_code": "np.apply_along_axis(lambda x:x**2,axis=0,arr=np.arange(1,6))",
        "example": lambda: np.apply_along_axis(lambda x:x**2,axis=0,arr=np.arange(1,6))
    },
    "np.char.upper": {
        "syntax": "np.char.upper(a)",
        "description": "Converts all characters in each string to uppercase.",
        "example_code": 'np.char.upper(np.array(["raghu","Rohan","RoHit","Rahul"]))',
        "example": lambda: np.char.upper(np.array(["raghu","Rohan","RoHit","Rahul"]))
    },
    "np.char.lower": {
        "syntax": "np.char.upper(a)",
        "description": "Converts all characters in each string to lowercase",
        "example_code": 'np.char.lower(np.array(["raghu","Rohan","RoHit","Rahul"]))',
        "example": lambda: np.char.lower(np.array(["raghu","Rohan","RoHit","Rahul"]))
    },
    "np.char.title": {
        "syntax": "np.char.title(a)",
        "description": "Converts the first character of each word to uppercase and all other characters to lowercase.",
        "example_code": 'np.char.title(np.array(["raghu","Rohan","RoHit","Rahul"]))',
        "example": lambda: np.char.title(np.array(["raghu","Rohan","RoHit","Rahul"]))
    },
    "np.char.capitalize": {
        "syntax": "np.char.title(a)",
        "description": "Converts the first character of each string to uppercase and all other characters to lowercase",
        "example_code": 'np.char.capitalize(np.array(["raghu","Rohan","RoHit","Rahul"]))',
        "example": lambda: np.char.capitalize(np.array(["raghu","Rohan","RoHit","Rahul"]))
    },
    "np.char.split": {
        "syntax": "np.char.split(a,sep=None)",
        "description": "Splits each string into a list of substrings based on a delimiter",
        "example_code": 'np.char.split(np.array([" Python "," SQL "," ML "," DL "]),sep=" ")',
        "example": lambda: np.char.split(np.array([" Python "," SQL "," ML "," DL "]),sep=" ")
    
    },
    "np.char.strip": {
        "syntax": "np.char.strip(a)",
        "description": "Removes leading and trailing whitespace from each string.",
        "example_code": 'np.char.strip(np.array([" Python "," SQL "," ML "," DL "]))',
        "example": lambda: np.char.strip(np.array([" Python "," SQL "," ML "," DL "]))
    },
    "np.char.join": {
        "syntax": "np.char.join(sep,a)",
        "description": "Joins elements of a list of strings with a specified delimiter",
        "example_code": 'np.char.join("_",np.array([" Python "," SQL "," ML "," DL "]))',
        "example": lambda: np.char.join("_",np.array([" Python "," SQL "," ML "," DL "]))
    },
    "np.exp": {
        "syntax": "np.exp(a)",
        "description": "This function computes the exponential of each element in the input array x. The exponential function calculates � � e x , where � e is Euler's number (approximately 2.71828)",
        "example_code": 'np.exp(np.array[1,3,4,6])',
        "example": lambda: np.exp(np.array([1,3,4,6]))
    },
    "np.std": {
        "syntax": "np.std(a)",
        "description": "This function computes the standard deviation of the input array.",
        "example_code": 'np.std(np.array[1,3,4,6])',
        "example": lambda: np.std(np.array([1,3,4,6]))
    }
}

# Display the selected function's details and output
if selected_function:
    func_details = examples[selected_function]
    st.subheader(f"Function: {selected_function}")
    st.write(f"**Syntax:** `{func_details['syntax']}`")
    st.write(f"**Description:** {func_details['description']}")
    st.write(f"**Example Code:** `{func_details['example_code']}`")
    st.write("**Example Output:**")
    st.write(func_details["example"]())
