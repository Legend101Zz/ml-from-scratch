"""
This file implements the "instant solution" to linear regression using the Normal Equation.
Unlike gradient descent which iteratively improves weights, this calculates the optimal
weights directly in one mathematical operation!

MY JOURNEY WITH THIS:
--------------------
When I first learned about the Normal Equation, I thought "Wait, if there's a direct
solution, why did I spend all that time implementing gradient descent?!"

Then I implemented it and discovered the truth:
1. The Normal Equation is BEAUTIFUL mathematically
2. It's perfect for small datasets (< 10,000 samples)
3. But it BREAKS DOWN for large datasets (matrix inversion is expensive!)
4. It ONLY works for linear regression (no closed-form for logistic regression, neural nets)
5. Gradient descent is more general and scalable

So both approaches are valuable! The Normal Equation taught me that sometimes there ARE
shortcuts in math, but understanding why they work (and when they fail) requires knowing
the iterative approach first.

THE MATHEMATICS:
---------------
Starting from the loss function:
    J(w) = (1/n) ||Xw - y||²

We want to minimize this. Taking the derivative and setting it to zero:
    ∇J(w) = (2/n)Xᵀ(Xw - y) = 0
    
Solving for w:
    Xᵀ(Xw - y) = 0
    XᵀXw - Xᵀy = 0
    XᵀXw = Xᵀy
    w = (XᵀX)⁻¹Xᵀy

This is the Normal Equation! It gives the exact optimal weights in one calculation.

WHY IT'S CALLED "NORMAL":
------------------------
The term "normal" comes from the geometric interpretation: at the optimal solution,
the residual vector (y - Xw) is orthogonal (perpendicular, "normal") to the column
space of X. This is a beautiful result from linear algebra!

THE CATCH:
---------
Computing (XᵀX)⁻¹ requires matrix inversion, which is:
1. Computationally expensive: O(n³) where n is the number of features
2. Numerically unstable for ill-conditioned matrices
3. Impossible if XᵀX is singular (not invertible)

When does XᵀX become singular?
- When features are perfectly correlated (e.g., x₂ = 2x₁)
- When you have more features than samples (n_features > n_samples)
- When features have no variance (all same value)

In these cases, gradient descent still works, but Normal Equation fails!

WHEN TO USE WHICH:
-----------------
Normal Equation:
  ✓ Small datasets (< 10,000 samples)
  ✓ Want exact solution (no approximation)
  ✓ Don't want to tune hyperparameters
  ✗ Large datasets (matrix inversion too slow)
  ✗ Singular matrix problems
  ✗ Only works for linear regression

Gradient Descent:
  ✓ Any dataset size (scales linearly)
  ✓ Works for any differentiable loss function
  ✓ Can use online learning (process data as it arrives)
  ✗ Need to tune learning rate and epochs
  ✗ Only finds approximate solution
"""


from foundations.linear_algebra.vectors_and_matrices import Matrix, Vector


class LinearRegressionClosedForm:
    """
    Linear Regression using the Normal Equation (closed-form solution).
    
    MATHEMATICAL FORMULA:
    --------------------
    w = (XᵀX)⁻¹Xᵀy  

    PARAMETERS:
    ----------
    None! The beauty of the Normal Equation is that it has no hyperparameters.
    No learning rate to tune, no epochs to choose. It just works (when it works).
    
    ATTRIBUTES:
    ----------
    weights_ : Vector
        The optimal weights computed via the Normal Equation.
    """
    
    def __init__(self):
        # No hyperparameters needed!
        self.weights_ = None
        
    def fit(self,X:Matrix, y:Matrix) -> 'LinearRegressionClosedForm':
        """
        Compute optimal weights using the Normal Equation.
        """
        # Step 1: Validation
        if X.num_rows != y.num_rows:
            raise ValueError(
                f"X has {X.num_rows} samples but y has {y.num_rows} samples. "
                "They must match!"
            )
        
        if y.num_cols != 1:
            raise ValueError(
                f"y must be a column vector, got shape {y.shape}"
            )
        
        # Warn if we might have numerical issues
        if X.num_cols > X.num_rows:
            import warnings
            warnings.warn(
                f"More features ({X.num_cols}) than samples ({X.num_rows}). "
                "The Normal Equation might fail or give unstable results. "
                "Consider using gradient descent instead."
            )
            
        n_samples = X.num_rows
        n_features = X.num_cols
        
        # Compute X.T @ X manually since we don't have built-in transpose
        # XᵀX[i,j] = Σₖ X[k,i] × X[k,j]
        XtX_elements = []
        for i in range(n_features):
            row = []
            for j in range(n_features):
                # Dot Product of column i and column j
                col_i = X.column(i)
                col_j = X.column(j)
                element = col_i.dot(col_j)
                row.append(element)
            XtX_elements.append(row)
            
        XtX = Matrix(XtX_elements)
        
        # Step 3: Compute (XᵀX)⁻¹ (the hard part!)
        
        # Matrix inversion is expensive: O(n³) for an n×n matrix.
        # We'll implement a simple version using Gaussian elimination.
        #
        # For a 2×2 matrix [[a,b],[c,d]], the inverse is:
        # (1/det) × [[d,-b],[-c,a]] where det = ad-bc
        
        if n_features ==2:
            # Speical case : 2*2 matrix has closed form
            a, b = XtX[0,0] , XtX[0,1]
            c, d = XtX[1,0] , XtX[1,1]
            
            det = a*d - b*c
            
            if abs(det) < 1e-10:
                raise ValueError("Matrix is singular , cannot invert!")
            
            inv_det = 1.0 / det
            
            XtX_inv = Matrix([
                [ d * inv_det, -b * inv_det],
                [-c * inv_det,  a * inv_det] 
            ])
        else:
            # General case: Use Gaussian elimination with partial pivoting
            # This is the most numerically stable method for small matrices
            XtX_inv = self._invert_matrix(XtX)
            
        # Step 4: Compute Xᵀy
        # Xᵀy is a vector of length n_features
        # Element i is the dot product of column i of X with y
        
        Xty_elements = []
        for i in range(n_features):
            col_i = X.column(i)
            # We need to extract y as a vector for dot product
            y_vec = Vector([y[j,0] for j in range(n_samples)])
            element = col_i.dot(y_vec)
            Xty_elements.append(element)
            
        Xty = Vector(Xty_elements)
        
        # Step 5: Compute w = (XᵀX)⁻¹Xᵀy
        # This is the final step: multiply the inverse by Xᵀy
        # The result is our optimal weights!
        
        weights_elements = []
        for i in range(n_features):
            # Dot product of row i of (XᵀX)⁻¹ with Xᵀy
            row_i = XtX_inv.row(i)
            weight_i = row_i.dot(Xty)
            weights_elements.append(weight_i)
            
        self.weights_ = Vector(weights_elements)
        
        return self
    
    def predict(self, X: Matrix) -> Matrix:
        """
        Make predictions using the computed weights.
        
        This is identical to the gradient descent version: ŷ = Xw
        The only difference is HOW we computed w (Normal Equation vs. gradient descent).
        
        PARAMETERS:
        ----------
        X : Matrix
            Features to predict on (must include bias column)
            
        RETURNS:
        -------
        predictions : Matrix
            Predicted values
        """
        
        if self.weights_ is None:
            raise RuntimeError(
                "Model not trained! Call .fit(X, y) first."
            )
        
        if X.num_cols != len(self.weights_):
            raise ValueError(
                f"X has {X.num_cols} features but model expects {len(self.weights_)}"
            )
        
        # Compute ŷ = Xw
        predictions = []
        for i in range(X.num_rows):
            x_i = X.row(i)
            y_pred = x_i.dot(self.weights_)
            predictions.append([y_pred])
        
        return Matrix(predictions)
    
    def score(self, X: Matrix, y: Matrix) -> float:
        """
        Compute R² score.
        
        (Same implementation as gradient descent version)
        """
        y_pred = self.predict(X)
        
        y_values = [y[i, 0] for i in range(y.num_rows)]
        y_mean = sum(y_values) / len(y_values)
        
        ss_residual = sum((y[i, 0] - y_pred[i, 0]) ** 2 
                         for i in range(y.num_rows))
        ss_total = sum((y[i, 0] - y_mean) ** 2 
                      for i in range(y.num_rows))
        
        if ss_total == 0:
            return 1.0 if ss_residual == 0 else 0.0
        
        return 1.0 - (ss_residual / ss_total)
    
    def _invert_matrix(self, M: Matrix) -> Matrix:
        """
        Invert a matrix using Gaussian elimination with partial pivoting.
        
        This is a general-purpose matrix inversion algorithm that works for any
        invertible square matrix. It's not the fastest, but it's numerically stable
        and educational to implement!
        
        THE ALGORITHM:
        -------------
        We create an augmented matrix [M | I] and use row operations to transform
        it into [I | M⁻¹]. The identity matrix on the right becomes the inverse!
        
        Steps:
        1. Create augmented matrix [M | I]
        2. For each column:
           - Find the largest pivot (for numerical stability)
           - Swap rows to bring pivot to diagonal
           - Scale row so diagonal element is 1
           - Eliminate all other elements in that column
        3. Extract the right half (which is now M⁻¹)
        
        WHY PARTIAL PIVOTING:
        --------------------
        Choosing the largest pivot reduces numerical errors. Without pivoting,
        dividing by a very small number can cause huge rounding errors.
        
        PARAMETERS:
        ----------
        M : Matrix
            Square matrix to invert
            
        RETURNS:
        -------
        M_inv : Matrix
            Inverse of M
            
        RAISES:
        ------
        ValueError : If matrix is singular (not invertible)
        """
        
        n = M.num_rows
        
        if M.num_rows != M.num_cols:
            raise ValueError("Can only invert square matrices!")
        
        # Create augmented matrix [M | I]
        # We'll work with a copy to avoid modifying M
        aug = []
        for i in range(n):
            row = []
            # Copy M's row
            for j in range(n):
                row.append(float(M[i, j]))
            # Add identity matrix
            for j in range(n):
                row.append(1.0 if i == j else 0.0)
            aug.append(row)
        
        # Gaussian elimination with partial pivoting
        for col in range(n):
            # Find the row with largest absolute value in this column (pivot)
            max_row = col
            max_val = abs(aug[col][col])
            
            for row in range(col + 1, n):
                if abs(aug[row][col]) > max_val:
                    max_val = abs(aug[row][col])
                    max_row = row
            
            # Check if pivot is too small (matrix is singular)
            if abs(aug[max_row][col]) < 1e-10:
                raise ValueError(
                    f"Matrix is singular! Column {col} has no pivot. "
                    "Cannot invert."
                )
            
            # Swap rows if necessary
            if max_row != col:
                aug[col], aug[max_row] = aug[max_row], aug[col]
            
            # Scale pivot row so diagonal element is 1
            pivot = aug[col][col]
            for j in range(2 * n):
                aug[col][j] /= pivot
            
            # Eliminate all other elements in this column
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    for j in range(2 * n):
                        aug[row][j] -= factor * aug[col][j]
        
        # Extract the inverse from the right half of augmented matrix
        inv_elements = []
        for i in range(n):
            row = []
            for j in range(n, 2 * n):
                row.append(aug[i][j])
            inv_elements.append(row)
        
        return Matrix(inv_elements)
