"""
VECTORS AND MATRICES: THE BUILDING BLOCKS OF MACHINE LEARNING
==============================================================

This module implements the fundamental data structures of linear algebra: vectors and matrices.
These are not just mathematical abstractions - they are how we represent ALL data in machine learning.
-------------------
We will implement these from scratch using only Python lists, showing you that there is no magic.
Vectors and matrices are just organized collections of numbers with special operations defined on them.

By the end of this module, you will understand:
1. How to represent vectors and matrices in memory
2. What operations are fundamental and why
3. How these structures appear in real machine learning algorithms
4. The geometric meaning behind algebraic operations
"""

import math
from typing import List, Tuple, Union


class Vector: 
    """
        A Vector represents an ordered collection of numbers.
        
        MATHEMATICAL DEFINITION:
        -----------------------
        A vector in R^n (n-dimensional real space) is an ordered n-tuple of real numbers.
        For example, v = [1, 2, 3] is a vector in R^3 (3-dimensional space).
        
        GEOMETRIC INTERPRETATION:
        ------------------------
        You can think of a vector in two equivalent ways:
        
        1. AS AN ARROW: Starting from the origin, the vector points to a specific location.
        The vector [3, 4] means "go 3 units right, 4 units up from the origin."
        
        2. AS A POINT: The vector represents a point in space at coordinates (3, 4).
        This is useful when thinking about data: each data point is a vector!
        
        MACHINE LEARNING EXAMPLES:
        -------------------------
        - A house can be represented as a vector: [square_feet, bedrooms, age, distance_to_city]
        - An image pixel can be a vector: [red_intensity, green_intensity, blue_intensity]
        - A user's movie ratings form a vector: [rating_movie1, rating_movie2, ...]
        - Model parameters (weights) are stored in vectors
        
        WHY WE NEED THIS CLASS:
        ----------------------
        While Python lists can hold numbers, they don't have the mathematical operations
        we need for machine learning (dot products, norms, etc.). This class adds those operations.
        """
    def __init__(self, elements: List[float]):
        """
        Initialize a vector from a list of numbers.
        
        Parameters:
        ----------
        elements : List[float]
            The components of the vector. Order matters!
            
        Example:
        -------
        v = Vector([1, 2, 3])  # Creates vector [1, 2, 3] in R^3
        
        DESIGN CHOICE:
        -------------
        We store the elements in a Python list internally. This is simple but not the most
        efficient for large vectors. Professional libraries like NumPy use C arrays for speed.
        But for learning, lists are perfect because they're simple and we can see exactly
        what's happening.
        """
        self.elements = elements
        self.dimension = len(elements) # How many components does this vector have?
        
    def __repr__(self) -> str:
        """
        Return a string representation of the vector for printing.
        
        This "magic method" is called when you print a Vector object or view it in the console.
        Instead of seeing "<Vector object at 0x...>", you'll see something good and readable like "Vector([1, 2, 3])".
        """
        return f"Vector({self.elements})"

    def __len__(self) -> int:
        """
        Return the dimension (number of components) of the vector.
        
        This allows you to use len(v) just like with lists:
        >>> v = Vector([1, 2, 3])
        >>> len(v)
        3
        
        WHY THIS MATTERS:
        ----------------
        The dimension tells you what space the vector lives in. A vector with 3 components
        is a point in 3D space. A vector with 1000 components is a point in 1000-dimensional
        space (which we can't visualize, but the math works the same!).
        """
        return self.dimension
    
    def __getitem__(self,index: int) -> float:
        """
        Access individual components of the vector using bracket notation.
        
        This allows you to use v[0], v[1], etc., just like with lists:
        >>> v = Vector([1, 2, 3])
        >>> v[0]
        1
        >>> v[2]
        3
        
        INDEXING CONVENTION:
        -------------------
        We use 0-based indexing (Python convention): first element is index 0.
        In mathematics, vectors are often written with 1-based indexing: v_1, v_2, v_3.
        Be careful when translating between mathematical notation and code!
        """
        return self.elements[index]

    def __setitem__(self, index: int, value: float):
        """
        Modify individual components of the vector.
        
        This allows you to change components:
        >>> v = Vector([1, 2, 3])
        >>> v[1] = 5
        >>> v
        Vector([1, 5, 3])
        
        MUTABILITY CONSIDERATION:
        ------------------------
        This makes vectors mutable (changeable). In some applications, you might want
        immutable vectors to prevent accidental modifications. For our learning purposes,
        mutability is fine and makes the code simpler.
        """
        self.elements[index] = value
        
    def __eq__(self, other: 'Vector') -> bool:
        """
        Check if two vectors are equal (have the same components).
        
        This allows you to use == to compare vectors:
        >>> v1 = Vector([1, 2, 3])
        >>> v2 = Vector([1, 2, 3])
        >>> v1 == v2
        True
        
        MATHEMATICAL DEFINITION OF EQUALITY:
        -----------------------------------
        Two vectors are equal if and only if they have the same dimension and
        corresponding components are equal. Order matters: [1, 2] ≠ [2, 1].
        """
        if not isinstance(other,Vector):
            return False
        return self.elements == other.elements
    
    def __add__(self,other: 'Vector') -> 'Vector':
        """
        Add two vectors component-wise.
        
        This allows you to use + to add vectors:
        >>> v1 = Vector([1, 2, 3])
        >>> v2 = Vector([4, 5, 6])
        >>> v1 + v2
        Vector([5, 7, 9])
        
        MATHEMATICAL DEFINITION:
        -----------------------
        Vector addition is defined component-wise:
        [a1, a2, a3] + [b1, b2, b3] = [a1+b1, a2+b2, a3+b3]
        
        GEOMETRIC INTERPRETATION:
        ------------------------
        Imagine vectors as arrows. To add them, place the tail of the second arrow
        at the tip of the first arrow. The sum is the arrow from the origin to the
        tip of the second arrow. This is the "tip-to-tail" method.
        
        Example: You walk 3 blocks east and 4 blocks north (vector [3, 4]).
                 Then you walk 2 blocks east and 1 block north (vector [2, 1]).
                 Your total displacement is [3+2, 4+1] = [5, 5].
        
        WHY IT MUST BE COMPONENT-WISE:
        ------------------------------
        Any other definition would not preserve the geometric meaning of "displacement."
        Adding different components (like a1+b2) would produce nonsensical results.
        
        REQUIREMENT:
        -----------
        Both vectors must have the same dimension. You can't add a 2D vector to a 3D vector!
        """
        if self.dimension != other.dimension:
            raise ValueError(
                f"Cannot add vectors of different dimensions: "
                f"{self.dimension} and {other.dimension}. "
                f"They must have the same number of components."
            )
        
        # Create a new vector where each component is the sum of corresponding components
        result_elements = [self[i] + other[i] for i in range(self.dimension)]
        return Vector(result_elements)
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        """
        Subtract one vector from another component-wise.
        
        This allows you to use - to subtract vectors:
        >>> v1 = Vector([5, 7, 9])
        >>> v2 = Vector([1, 2, 3])
        >>> v1 - v2
        Vector([4, 5, 6])
        
        MATHEMATICAL DEFINITION:
        -----------------------
        Subtraction is addition of the negative:
        v1 - v2 = v1 + (-v2)
        
        GEOMETRIC INTERPRETATION:
        ------------------------
        v1 - v2 gives you the vector FROM v2 TO v1.
        If v1 is your final position and v2 is your starting position,
        then v1 - v2 is your displacement (how you moved).
        
        MACHINE LEARNING EXAMPLE:
        ------------------------
        In gradient descent, you compute: new_weights = old_weights - learning_rate * gradient
        The subtraction moves you in the opposite direction of the gradient (downhill).
        """
        if self.dimension != other.dimension:
            raise ValueError(
                f"Cannot subtract vectors of different dimensions: "
                f"{self.dimension} and {other.dimension}"
            )
        
        result_elements = [self[i] - other[i] for i in range(self.dimension)]
        return Vector(result_elements)

    def __mul__(self, scalar: float) -> 'Vector':
        """
        Multiply a vector by a scalar (scale the vector).
        
        This allows you to use * to scale vectors:
        >>> v = Vector([1, 2, 3])
        >>> v * 3
        Vector([3, 6, 9])
        
        MATHEMATICAL DEFINITION:
        -----------------------
        Scalar multiplication scales each component:
        c * [a1, a2, a3] = [c*a1, c*a2, c*a3]
        
        GEOMETRIC INTERPRETATION:
        ------------------------
        Multiplying by a positive scalar stretches the vector (makes it longer) without
        changing its direction. Multiplying by a negative scalar flips the direction.
        Multiplying by a value between 0 and 1 shrinks the vector.
        
        Examples:
        - v * 2: doubles the length, same direction
        - v * -1: same length, opposite direction
        - v * 0.5: halves the length, same direction
        
        MACHINE LEARNING EXAMPLE:
        ------------------------
        When you adjust weights in gradient descent, you multiply the gradient by the
        learning rate (a scalar). This controls how big a step you take:
        step = learning_rate * gradient
        """
        result_elements = [scalar * element for element in self.elements]
        return Vector(result_elements)

    def __rmul__(self, scalar: float) -> 'Vector':
        """
        Right multiplication: scalar * vector (order reversed).
        
        This allows: 3 * v  (in addition to v * 3)
        
        WHY WE NEED THIS:
        ----------------
        When you write "3 * v", Python tries to call (3).__mul__(v).
        But integers don't know how to multiply by Vector objects!
        So Python falls back to trying v.__rmul__(3).
        By defining __rmul__, we handle this case.
        
        COMMUTATIVITY:
        -------------
        Scalar multiplication is commutative: c * v = v * c
        So we just call the forward multiplication.
        """
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector':
        """
        Divide a vector by a scalar.
        
        This allows you to use / to scale vectors:
        >>> v = Vector([6, 9, 12])
        >>> v / 3
        Vector([2.0, 3.0, 4.0])
        
        MATHEMATICAL NOTE:
        -----------------
        Division by scalar c is the same as multiplication by 1/c.
        We implement it separately for convenience and clarity.
        
        IMPORTANT: Division by zero is undefined and will raise an error!
        """
        if scalar == 0:
            raise ValueError("Cannot divide vector by zero!")
        
        # Division is multiplication by the reciprocal
        return self * (1.0 / scalar)
    
    def norm(self,p: int=2) -> float:
        """
        A norm is simply a mathematical function that tells you how long or big a vector is.
        To Compute the p-norm (length) of the vector.
        
        Parameters:
        ----------
        p : int
            Which norm to compute. Common values:
            - p=1: Manhattan norm (sum of absolute values)
            - p=2: Euclidean norm (ordinary distance, default)
            - p=∞: Maximum norm (largest absolute value)
        
        Returns:
        -------
        float : The norm (length) of the vector
        
        MATHEMATICAL DEFINITION:
        -----------------------
        The p-norm is: ||v||_p = (|v1|^p + |v2|^p + ... + |vn|^p)^(1/p)
        
        For p=2 (Euclidean norm): ||v|| = sqrt(v1^2 + v2^2 + ... + vn^2)
        This is the generalization of the Pythagorean theorem!
        
        GEOMETRIC INTERPRETATION:
        ------------------------
        The norm measures the LENGTH of the vector arrow.
        For v = [3, 4], the Euclidean norm is sqrt(3^2 + 4^2) = 5.
        This is the distance from the origin to the point (3, 4).
        
        WHY DIFFERENT NORMS EXIST:
        -------------------------
        Different norms measure "size" in different ways:
        
        - L1 (Manhattan): Distance if you can only move along grid lines.
          Example: Walking in a city with rectangular blocks.
          
        - L2 (Euclidean): Straight-line distance ("as the crow flies").
          Example: Distance a bird would fly between two points.
          
        - L∞ (Maximum): The largest component dominates.
          Example: Time to complete tasks (bottleneck is the longest task).
        
        MACHINE LEARNING USES:
        ---------------------
        - Measuring prediction errors (L1 and L2 loss)
        - Regularization (penalizing large weights using L1 or L2 norm)
        - Computing distances between data points in KNN
        - Normalizing vectors to unit length
        
        Examples:
        --------
        >>> v = Vector([3, 4])
        >>> v.norm(p=2)  # Euclidean: sqrt(9 + 16) = 5
        5.0
        >>> v.norm(p=1)  # Manhattan: |3| + |4| = 7
        7.0
        >>> v.norm(p=float('inf'))  # Maximum: max(|3|, |4|) = 4
        4.0
        """
        if p == float('inf'):
            # Maximum norm: largest absolute value
            return max(abs(element) for element in self.elements)
        elif p == 1:
            # Manhattan norm: sum of absolute values
            return sum(abs(element) for element in self.elements)
        elif p ==2:
            # Euclidean norm : most common , so we optimise it
            return math.sqrt(sum(element**2 for element in self.elements))
        else:
            # General p-norm
            return sum(abs(element) ** p for element in self.elements) ** (1.0 /p)
        
    def normalize(self) -> 'Vector':
        """
        Return a unit vector in the same direction (norm = 1).
        
        Returns:
        -------
        Vector : A new vector with the same direction but length 1
        
        MATHEMATICAL DEFINITION:
        -----------------------
        The unit vector in direction v is: v_hat = v / ||v||
        where ||v|| is the norm (length) of v.
        
        GEOMETRIC INTERPRETATION:
        ------------------------
        Normalization preserves direction but makes the length exactly 1.
        Think of it as "pointing in the same direction but with standard length."
        
        WHY THIS IS USEFUL:
        ------------------
        Unit vectors represent pure direction without magnitude.
        
        MACHINE LEARNING USES:
        ---------------------
        - In cosine similarity, you normalize vectors to compare directions only
        - In neural networks, weight initialization often uses normalized vectors
        - In gradient descent, the gradient direction matters, not just magnitude
        - In data preprocessing, feature normalization can use unit vectors
        
        IMPORTANT CASE:
        --------------
        The zero vector [0, 0, ..., 0] has no direction and cannot be normalized!
        Attempting to normalize it would require division by zero.
        
        Example:
        -------
        >>> v = Vector([3, 4])
        >>> v.norm()
        5.0
        >>> v_unit = v.normalize()
        >>> v_unit
        Vector([0.6, 0.8])
        >>> v_unit.norm()
        1.0
        """
        norm = self.norm()
        
        if norm == 0:
            raise ValueError(
                "Cannot normalize the zero vector! "
                "It has no direction (all components are zero)."
            )
        
        # Divide every component by the norm 
        return self / norm  
    
    def dot(self, other: 'Vector') -> float:
        """
        Compute the dot product (inner product, scalar product) with another vector.
        
        Parameters:
        ----------
        other : Vector
            The vector to take the dot product with
        
        Returns:
        -------
        float : The dot product (a scalar, not a vector!)
        
        MATHEMATICAL DEFINITION:
        -----------------------
        For vectors a and b:
        a · b = a1*b1 + a2*b2 + ... + an*bn
        
        Simply multiply corresponding components and sum them up.
        
        GEOMETRIC INTERPRETATION:
        ------------------------
        The dot product is related to the angle θ between vectors:
        a · b = ||a|| * ||b|| * cos(θ)
        
        This means:
        - If a and b point in the same direction (θ = 0°): dot product is maximum
        - If a and b are perpendicular (θ = 90°): dot product is zero
        - If a and b point opposite ways (θ = 180°): dot product is negative
        
        WHY THIS IS FUNDAMENTAL:
        -----------------------
        The dot product measures ALIGNMENT. It answers: "How much do these vectors
        point in the same direction?"
        
        MACHINE LEARNING USES (EVERYWHERE!):
        ----------------------------------
        1. Linear Regression: prediction = weights · features
        2. Neural Networks: each neuron computes a dot product of inputs and weights
        3. Cosine Similarity: similarity = (a · b) / (||a|| * ||b||)
        4. Attention Mechanisms: attention scores are dot products
        5. Support Vector Machines: decision boundary uses dot products
        
        PROJECTION INTERPRETATION:
        -------------------------
        (a · b) / ||b|| gives the length of the projection of a onto b.
        This measures "how much of a is in the direction of b."
        
        Example:
        -------
        >>> a = Vector([1, 2, 3])
        >>> b = Vector([4, 5, 6])
        >>> a.dot(b)
        32.0  # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        
        Perpendicular example:
        >>> x = Vector([1, 0])
        >>> y = Vector([0, 1])
        >>> x.dot(y)
        0.0  # Perpendicular vectors have dot product zero!
        """
        if self.dimension != other.dimension:
            raise ValueError(
                f"Cannot take dot product of vectors with different dimensions: "
                f"{self.dimension} and {other.dimension}"
            )
        
        # Multiply corresponding components and sum
        return sum(self[i] * other[i] for i in range(self.dimension))
    
    def angle_with(self, other: 'Vector') -> float:
        """
        Compute the angle (in radians) between this vector and another.
        
        Parameters:
        ----------
        other : Vector
            The vector to measure the angle to
        
        Returns:
        -------
        float : Angle in radians (0 to π)
        
        MATHEMATICAL FORMULA:
        --------------------
        From the geometric interpretation of dot product:
        cos(θ) = (a · b) / (||a|| * ||b||)
        
        Therefore:
        θ = arccos((a · b) / (||a|| * ||b||))
        
        INTERPRETATION:
        --------------
        - θ = 0: vectors point in exactly the same direction (parallel)
        - θ = π/2 (90°): vectors are perpendicular (orthogonal)
        - θ = π (180°): vectors point in opposite directions (antiparallel)
        
        MACHINE LEARNING USE:
        --------------------
        In document similarity, the angle between word frequency vectors
        indicates how similar two documents are. Small angle = similar content.
        
        Example:
        -------
        >>> v1 = Vector([1, 0])
        >>> v2 = Vector([0, 1])
        >>> import math
        >>> v1.angle_with(v2)
        1.5707963...  # π/2 radians = 90 degrees
        >>> math.degrees(v1.angle_with(v2))
        90.0
        """
        dot_product = self.dot(other)
        norm_product = self.norm() * other.norm()
        
        if norm_product == 0:
            raise ValueError(
                "Cannot compute angle involving zero vector! "
                "Zero vector has no direction."
            )
        
        # cos(θ) = dot / (norm1 * norm2)
        # Clamp to [-1, 1] to handle numerical errors
        cos_angle = max(-1.0, min(1.0, dot_product / norm_product))
        
        return math.acos(cos_angle)
    
class Matrix:
    """
    A Matrix represents a 2D array of numbers arranged in rows and columns.
    
    MATHEMATICAL DEFINITION:
    -----------------------
    A matrix is a rectangular array of numbers with m rows and n columns.
    We say the matrix is m-by-n (rows by columns).
    
    Example of a 3-by-2 matrix:
    [[1, 2],
     [3, 4],
     [5, 6]]
    
    This has 3 rows and 2 columns, so it's 3-by-2.
    
    GEOMETRIC INTERPRETATION:
    ------------------------
    A matrix represents a LINEAR TRANSFORMATION of space.
    When you multiply a vector by a matrix, you transform that vector.
    
    The transformation can:
    - Rotate the vector
    - Scale it (stretch or shrink)
    - Reflect it (flip it)
    - Project it to lower dimensions
    - Or any combination of these!
    
    MACHINE LEARNING EXAMPLES:
    -------------------------
    - DATA MATRIX: Each row is a data point, each column is a feature
      [[house1_sqft, house1_bedrooms, house1_age],
       [house2_sqft, house2_bedrooms, house2_age],
       ...]
    
    - WEIGHT MATRIX: In neural networks, weights are arranged in matrices
      Each row connects one layer's neurons to the next layer
    
    - COVARIANCE MATRIX: Shows how features vary together
      Element (i,j) is the covariance between feature i and feature j
    
    - TRANSFORMATION MATRIX: In PCA, the principal components form a matrix
      Multiplying data by this matrix projects it to lower dimensions
    
    WHY WE NEED THIS CLASS:
    ----------------------
    Like vectors, Python's nested lists can store matrices but don't have
    the mathematical operations (multiplication, transpose, etc.) we need.
    This class adds those operations.
    """
    def __init__(self,elements: List[List[float]]):
        """
        Initialize a matrix from a list of lists (rows).
        
        Parameters:
        ----------
        elements : List[List[float]]
            A list of rows, where each row is a list of numbers
            
        Example:
        -------
        m = Matrix([[1, 2, 3],
                    [4, 5, 6]])
        This creates a 2-by-3 matrix (2 rows, 3 columns).
        
        INTERNAL REPRESENTATION:
        -----------------------
        We store the matrix as a list of rows. Each row is a list of numbers.
        This is called "row-major" order and is the most natural representation.
        
        IMPORTANT REQUIREMENT:
        ---------------------
        All rows must have the same length! A matrix must be rectangular.
        You cannot have rows of different lengths.
        """
        if not elements:
            raise ValueError("Matrix cannot be empty")
        
        # Check that all rows have the same length
        row_lengths = [len(row) for row in elements]
        if len(set(row_lengths)) > 1:
            raise ValueError(
                f"All rows must have the same length! "
                f"Got row lengths: {row_lengths}"
            )
        
        self.elements = elements
        self.num_rows = len(elements)
        self.num_cols = len(elements[0])
        self.shape = (self.num_cols,self.num_cols)

    def __repr__(self) -> str:
        """Return a readable string representation of the matrix."""
        return f"Matrix({self.elements})"
    
    def __getitem__(self, index: Tuple[int, int]) -> float:
        """
        Access individual elements using [row, col] notation.
        
        Example:
        -------
        >>> m = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> m[0, 1]
        2
        >>> m[1, 2]
        6
        
        INDEXING CONVENTION:
        -------------------
        We use [row, column] indexing, both starting from 0.
        In mathematics, this element is often written as M_(i+1, j+1)
        (1-based indexing).
        """
        row, column = index
        return self.elements[row][column]

    def __setitem__(self, index: Tuple[int, int], value: float):
        """
        Modify individual elements using [row, col] notation.
        
        Example:
        -------
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> m[0, 1] = 10
        >>> m
        Matrix([[1, 10], [3, 4]])
        """
        row, col = index
        self.elements[row][col] = value
        
    def row(self, i:int) -> Vector:
        """
        Extract row i as a Vector.
        
        This is useful for operations that process rows independently.
        
        Example:
        -------
        >>> m = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> m.get_row(0)
        Vector([1, 2, 3])
        """
        return Vector(self.elements[i])
    
    def column(self, j:int) -> Vector:
        """
        Extract column j as a Vector.
        
        This requires gathering elements from each row.
        
        Example:
        -------
        >>> m = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> m.get_col(1)
        Vector([2, 5])
        """
        return Vector([self.elements[i][j] for i in range(self.num_rows)])
    
if __name__ == "__main__":
    print("=" * 70)
    print("VECTORS AND MATRICES: INTERACTIVE TUTORIAL")
    print("=" * 70)
    
    print("\n### CREATING VECTORS ###\n")
    
    # Create some vectors
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"Dimension of v1: {len(v1)}")
    
    print("\n### VECTOR OPERATIONS ###\n")
    
    # Addition
    v_sum = v1 + v2
    print(f"v1 + v2 = {v_sum}")
    
    # Scalar multiplication
    v_scaled = v1 * 3
    print(f"3 * v1 = {v_scaled}")
    
    # Norm (length)
    print(f"||v1|| (Euclidean norm) = {v1.norm():.4f}")
    print(f"||v1||_1 (Manhattan norm) = {v1.norm(p=1):.4f}")
    
    # Normalization
    v1_unit = v1.normalize()
    print(f"v1 normalized: {v1_unit}")
    print(f"Norm of normalized v1: {v1_unit.norm():.4f}")
    
    # Dot product
    dot = v1.dot(v2)
    print(f"v1 · v2 = {dot:.4f}")
    
    # Angle
    angle_rad = v1.angle_with(v2)
    angle_deg = math.degrees(angle_rad)
    print(f"Angle between v1 and v2: {angle_deg:.2f} degrees")
    
    print("\n### CREATING MATRICES ###\n")
    
    m1 = Matrix([[1, 2, 3],
                 [4, 5, 6]])
    
    print(f"m1 = {m1}")
    print(f"Shape: {m1.shape} (rows × columns)")
    print(f"Element at [0, 1]: {m1[0, 1]}")
    
    print("\n### EXTRACTING ROWS AND COLUMNS ###\n")
    
    row0 = m1.row(0)
    col1 = m1.column(1)
    
    print(f"Row 0: {row0}")
    print(f"Column 1: {col1}")
    
    print("\n✅ Vector and Matrix basics complete!")
    print("\nNext: Implement dot_products.py for advanced vector operations")
    print("Then: matrix_multiplication.py to see how matrices transform space")




