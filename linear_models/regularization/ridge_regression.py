"""
RIDGE REGRESSION: L2 Regularization to Prevent Overfitting
==========================================================

This module implements Ridge Regression, which adds an L2 penalty to linear regression
to prevent overfitting. When I first encountered this, I was skeptical: "Why would we
intentionally make our loss function worse?" Then I saw it dramatically improve test
performance, and everything clicked.

MY LEARNING JOURNEY:
-------------------
I had a dataset with 20 features predicting house prices. Standard linear regression
gave me R² = 0.98 on training, but only 0.65 on test data. Classic overfitting!

The weights were ridiculous: one feature had weight 10000, another had -8000. The model
was so sensitive to tiny changes in features that it fell apart on new data.

Then I added Ridge regularization (α=1.0):
- Training R²: dropped to 0.92 (slightly worse)
- Test R²: jumped to 0.88 (much better!)
- Weights: all reasonable magnitudes (between -5 and 5)

That's when I understood: Ridge doesn't make the model worse, it makes it more robust.

THE BIG INSIGHT:
---------------
Ridge regression solves the problem of models being "too confident" about weights.
Without regularization, the model might learn: "Feature A is EXTREMELY important (weight=1000)
and feature B is EXTREMELY negative (weight=-800)." These extreme weights memorize the
training data's quirks.

Ridge says: "I'll penalize you for having large weights. If you want a large weight,
it better be because that feature is REALLY important, not just because it fits the
training data slightly better."

THE MATHEMATICS:
---------------
Standard linear regression minimizes: Loss = (1/n) Σ(ŷᵢ - yᵢ)²

Ridge regression minimizes: Loss = (1/n) Σ(ŷᵢ - yᵢ)² + α Σwᵢ²
                                  = MSE + α × (w₁² + w₂² + ... + wₙ²)

The second term (α Σwᵢ²) is the L2 penalty:
- It grows quadratically with weight magnitude
- Small weights have small penalty
- Large weights have huge penalty (quadratic!)
- Forces weights to be small unless absolutely necessary

WHY "L2"? Because we're penalizing the L2 norm (Euclidean length) of the weight vector:
||w||₂ = √(w₁² + w₂² + ... + wₙ²)

Actually we penalize the squared L2 norm (||w||₂²) for mathematical convenience.

THE REGULARIZATION PARAMETER α:
------------------------------
α controls how much we care about small weights versus fitting the data:

- α = 0: No regularization (standard linear regression, might overfit)
- α small (0.001-0.1): Light regularization (slight smoothing)
- α medium (0.1-10): Moderate regularization (good for most cases)
- α large (10-1000): Strong regularization (very smooth, might underfit)

Choosing α is crucial and requires cross-validation!

THE CLOSED-FORM SOLUTION:
-------------------------
Ridge has a beautiful closed-form solution (unlike Lasso):

w = (XᵀX + αI)⁻¹Xᵀy

Where I is the identity matrix. Compare to regular linear regression:

w = (XᵀX)⁻¹Xᵀy

The addition of αI has multiple benefits:
1. Adds the L2 penalty mathematically
2. Makes XᵀX invertible (even if features are correlated!)
3. Improves numerical stability

This is where the name "Ridge" comes from - we're adding a "ridge" to the diagonal
of XᵀX, making it better conditioned.

WHY RIDGE HELPS WITH COLLINEARITY:
----------------------------------
When features are highly correlated (e.g., square footage in meters vs feet), XᵀX
becomes nearly singular and (XᵀX)⁻¹ is unstable. Adding αI stabilizes it!

Geometrically, collinear features create a "valley" in the loss surface where many
weight combinations give similar loss. Ridge prefers solutions with smaller weights
in this valley.

IMPORTANT: STANDARDIZE FIRST!
-----------------------------
Ridge penalizes all weights equally, but if features have different scales, this
is unfair! A feature measured in thousands naturally needs a small weight. A feature
measured in decimals needs a large weight. Same information, different penalty!

Always standardize features (mean=0, std=1) before applying Ridge, so the penalty
is fair across all features.

"""

from foundations.linear_algebra.vectors_and_matrices import Matrix, Vector


class RidgeRegression:
    """
    Ridge Regression with L2 regularization.
    
    This implements linear regression with an L2 penalty on the weights, preventing
    overfitting by discouraging large weight values.
    
    THE OBJECTIVE FUNCTION:
    ----------------------
    Ridge minimizes: J(w) = (1/n) Σ(ŷᵢ - yᵢ)² + α Σwⱼ²
    
    Where:
    - First term: Mean squared error (fit the data)
    - Second term: L2 penalty (keep weights small)
    - α: Regularization strength (tradeoff between terms)
    
    THE GRADIENT:
    ------------
    Taking the derivative with respect to w:
    
    ∇J(w) = (2/n) Xᵀ(Xw - y) + 2αw
          = (2/n) Xᵀ(ŷ - y) + 2αw
    
    The first part is the standard linear regression gradient.
    The second part (2αw) is the L2 penalty gradient - it pushes weights toward zero!
    
    IMPLEMENTATION CHOICES:
    ----------------------
    We implement Ridge using gradient descent rather than the closed-form solution
    because:
    1. Educational: Shows how regularization affects gradients
    2. Scalable: Works for large datasets (closed-form requires matrix inversion)
    3. Flexible: Can use different optimizers (SGD, Mini-Batch)
    4. Generalizable: Same approach works for other penalties (L1, Elastic Net)
    
    For small datasets, closed-form is faster. See RidgeRegressionClosedForm below.
    
    BIAS TERM HANDLING:
    ------------------
    IMPORTANT: We do NOT regularize the bias term (w₀)! Why?
    
    The bias just shifts predictions up/down - it doesn't affect model complexity.
    Regularizing it would make the model prefer predictions near zero, which is arbitrary.
    
    In practice: We regularize w₁, w₂, ..., wₙ but not w₀.
    
    PARAMETERS:
    ----------
    alpha : float, default=1.0
        Regularization strength. Must be positive.
        - Larger values = stronger regularization = simpler model
        - Smaller values = weaker regularization = more complex model
        
    learning_rate : float, default=0.01
        Step size for gradient descent
        
    n_epochs : int, default=100
        Number of training iterations
        
    optimizer : str, default='batch'
        Gradient descent variant to use
        
    verbose : bool, default=False
        Whether to print training progress
        
    ATTRIBUTES:
    ----------
    weights_ : Vector
        Learned weights (including bias as first element)
        
    history_ : dict
        Training history (loss per epoch)
        
    EXAMPLE USAGE:
    -------------
    >>> from foundations.data_preprocessing.feature_engineering import (
    ...     Standardizer, add_bias_column, train_test_split
    ... )
    >>> 
    >>> # Split data
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> 
    >>> # Standardize (CRITICAL for Ridge!)
    >>> scaler = Standardizer()
    >>> X_train_scaled = scaler.fit_transform(X_train)
    >>> X_test_scaled = scaler.transform(X_test)
    >>> 
    >>> # Add bias
    >>> X_train_final = add_bias_column(X_train_scaled)
    >>> X_test_final = add_bias_column(X_test_scaled)
    >>> 
    >>> # Train Ridge regression
    >>> model = RidgeRegression(alpha=1.0, n_epochs=200)
    >>> model.fit(X_train_final, y_train)
    >>> 
    >>> # Evaluate
    >>> print(f"Training R²: {model.score(X_train_final, y_train):.4f}")
    >>> print(f"Test R²: {model.score(X_test_final, y_test):.4f}")
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        optimizer: str = 'batch',
        verbose: bool = False
    ):
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.optimizer_type = optimizer
        self.verbose = verbose
        
        # Will be set during training
        self.weights_ = None
        self.history_ = None
    
    def fit(self, X: Matrix, y: Matrix) -> 'RidgeRegression':
        """
        Train Ridge regression model using gradient descent.
        
        This implements the iterative optimization of the Ridge objective function.
        We start with random (zero) weights and repeatedly compute gradients and
        update weights until convergence.
        
        THE TRAINING LOOP:
        -----------------
        For each epoch:
            1. Compute predictions: ŷ = Xw
            2. Compute errors: (ŷ - y)
            3. Compute gradient: ∇J = (1/n)Xᵀ(ŷ - y) + αw
            4. Update weights: w := w - learning_rate × ∇J
            5. Compute loss (for monitoring)
        
        THE GRADIENT COMPUTATION:
        ------------------------
        Standard regression gradient: (1/n) Xᵀ(ŷ - y)
        Plus L2 penalty gradient: αw
        
        The penalty gradient is simple: it's proportional to w itself!
        This means: "Whatever your weight is, push it toward zero by αw."
        
        Larger weights get pushed harder (linear with magnitude).
        Direction: always toward zero (if w positive, push negative; if negative, push positive).
        
        BIAS TERM SPECIAL CASE:
        ----------------------
        We modify the gradient to NOT regularize the bias (first weight).
        
        Standard gradient: ∇J = (1/n)Xᵀ(ŷ - y) + αw
        Modified gradient: ∇J = (1/n)Xᵀ(ŷ - y) + α[0, w₁, w₂, ..., wₙ]ᵀ
        
        The first element of the penalty is 0 (don't penalize bias).
        
        PARAMETERS:
        ----------
        X : Matrix, shape (n_samples, n_features)
            Training features (should include bias column and be standardized!)
            
        y : Matrix, shape (n_samples, 1)
            Training targets
            
        RETURNS:
        -------
        self : RidgeRegression
            Returns self for method chaining
        """
        
        if X.num_rows != y.num_rows:
            raise ValueError(
                f"X has {X.num_rows} samples but y has {y.num_rows} samples"
            )
        
        if y.num_cols != 1:
            raise ValueError(f"y must be column vector, got shape {y.shape}")
        
        n_samples = X.num_rows
        n_features = X.num_cols
        
        # Initialize weights to zeros
        self.weights_ = Vector([0.0] * n_features)
        
        # Training history
        self.history_ = {
            'loss': [],
            'epoch': []
        }
        
        if self.verbose:
            print("=" * 70)
            print("TRAINING RIDGE REGRESSION")
            print("=" * 70)
            print(f"Regularization strength (α): {self.alpha}")
            print(f"Dataset: {n_samples} samples, {n_features} features")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Epochs: {self.n_epochs}")
            print("-" * 70)
        
        # Training loop
        for epoch in range(self.n_epochs):
            # =====================================================================
            # STEP 1: Compute predictions
            # =====================================================================
            predictions = []
            for i in range(n_samples):
                x_i = X.row(i)
                y_pred_i = x_i.dot(self.weights_)
                predictions.append([y_pred_i])
            
            y_pred = Matrix(predictions)
            
            # =====================================================================
            # STEP 2: Compute loss (MSE + L2 penalty)
            # =====================================================================
            # MSE term
            mse = sum((y_pred[i, 0] - y[i, 0]) ** 2 for i in range(n_samples)) / n_samples
            
            # L2 penalty term (don't penalize bias!)
            l2_penalty = sum(self.weights_[j] ** 2 for j in range(1, n_features))
            
            # Total loss
            loss = mse + self.alpha * l2_penalty
            
            # =====================================================================
            # STEP 3: Compute gradient
            # =====================================================================
            # Standard gradient: (1/n) Xᵀ(ŷ - y)
            gradient_elements = []
            for j in range(n_features):
                feature_col = X.column(j)
                
                # Dot product with errors
                grad_j = 0.0
                for i in range(n_samples):
                    error = y_pred[i, 0] - y[i, 0]
                    grad_j += feature_col[i] * error
                
                grad_j /= n_samples
                
                # Add L2 penalty gradient (except for bias term)
                if j > 0:  # Don't regularize bias (j=0)
                    grad_j += self.alpha * self.weights_[j]
                
                gradient_elements.append(grad_j)
            
            gradient = Vector(gradient_elements)
            
            # =====================================================================
            # STEP 4: Update weights
            # =====================================================================
            # w := w - learning_rate × ∇J
            updated_weights = []
            for j in range(n_features):
                new_weight = self.weights_[j] - self.learning_rate * gradient[j]
                updated_weights.append(new_weight)
            
            self.weights_ = Vector(updated_weights)
            
            # =====================================================================
            # STEP 5: Record history
            # =====================================================================
            self.history_['loss'].append(loss)
            self.history_['epoch'].append(epoch)
            
            # Print progress
            if self.verbose and (epoch % 10 == 0 or epoch == self.n_epochs - 1):
                print(f"Epoch {epoch:4d}: Loss = {loss:.6f} "
                      f"(MSE = {mse:.6f}, L2 penalty = {self.alpha * l2_penalty:.6f})")
        
        if self.verbose:
            print("-" * 70)
            print("TRAINING COMPLETE!")
            print(f"Final loss: {self.history_['loss'][-1]:.6f}")
            print(f"Weight magnitudes: min={min(abs(w) for w in self.weights_.elements):.4f}, "
                  f"max={max(abs(w) for w in self.weights_.elements):.4f}")
            print("=" * 70)
        
        return self
    
    def predict(self, X: Matrix) -> Matrix:
        """
        Make predictions using learned weights.
        
        This is identical to standard linear regression: ŷ = Xw
        The regularization only affects training, not prediction!
        
        PARAMETERS:
        ----------
        X : Matrix
            Features to predict on (must have same format as training data)
            
        RETURNS:
        -------
        predictions : Matrix
            Predicted values
        """
        
        if self.weights_ is None:
            raise RuntimeError("Model not trained! Call .fit() first.")
        
        if X.num_cols != len(self.weights_):
            raise ValueError(
                f"X has {X.num_cols} features but model expects {len(self.weights_)}"
            )
        
        predictions = []
        for i in range(X.num_rows):
            x_i = X.row(i)
            y_pred = x_i.dot(self.weights_)
            predictions.append([y_pred])
        
        return Matrix(predictions)
    
    def score(self, X: Matrix, y: Matrix) -> float:
        """
        Compute R² score.
        
        PARAMETERS:
        ----------
        X : Matrix
            Features
        y : Matrix
            True targets
            
        RETURNS:
        -------
        r2 : float
            R² score
        """
        
        y_pred = self.predict(X)
        
        # Compute R²
        y_values = [y[i, 0] for i in range(y.num_rows)]
        y_mean = sum(y_values) / len(y_values)
        
        ss_residual = sum((y[i, 0] - y_pred[i, 0]) ** 2 for i in range(y.num_rows))
        ss_total = sum((y[i, 0] - y_mean) ** 2 for i in range(y.num_rows))
        
        if ss_total < 1e-10:
            return 1.0 if ss_residual < 1e-10 else 0.0
        
        return 1.0 - (ss_residual / ss_total)


class RidgeRegressionClosedForm:
    """
    Ridge Regression using the closed-form solution.
    
    For small to medium datasets, we can solve Ridge exactly using matrix algebra:
    
    w = (XᵀX + αI)⁻¹Xᵀy
    
    Where I is the identity matrix. This is instant (no iterative training!) but
    requires matrix inversion which is expensive for large datasets.
    
    THE MATHEMATICS:
    ---------------
    Starting from the Ridge objective:
    J(w) = ||Xw - y||² + α||w||²
    
    Taking the derivative and setting to zero:
    ∇J(w) = 2Xᵀ(Xw - y) + 2αw = 0
    XᵀXw - Xᵀy + αw = 0
    (XᵀX + αI)w = Xᵀy
    w = (XᵀX + αI)⁻¹Xᵀy
    
    THE BENEFITS OF αI:
    ------------------
    Adding αI to XᵀX has multiple benefits:
    
    1. **Regularization**: Mathematically encodes the L2 penalty
    2. **Invertibility**: Even if XᵀX is singular, (XᵀX + αI) is invertible!
    3. **Numerical stability**: Improves condition number of the matrix
    4. **Collinearity handling**: Stabilizes solution when features are correlated
    
    WHEN TO USE CLOSED-FORM VS GRADIENT DESCENT:
    -------------------------------------------
    Closed-form:
    - ✓ Exact solution (no approximation)
    - ✓ Fast for small datasets (< 10,000 samples, < 1,000 features)
    - ✓ No hyperparameters to tune (no learning rate, no epochs)
    - ✗ Expensive for large datasets (matrix inversion is O(n³))
    - ✗ Not suitable for online learning
    
    Gradient descent:
    - ✓ Scales to large datasets
    - ✓ Can use online/mini-batch learning
    - ✓ Works with other optimizers (momentum, Adam, etc.)
    - ✗ Requires tuning learning rate and epochs
    - ✗ Approximate solution (converges to optimum)
    
    EXAMPLE USAGE:
    -------------
    >>> model = RidgeRegressionClosedForm(alpha=1.0)
    >>> model.fit(X_train, y_train)  # Instant!
    >>> 
    >>> predictions = model.predict(X_test)
    >>> print(f"R² score: {model.score(X_test, y_test):.4f}")
    """
    
    def __init__(self, alpha: float = 1.0):
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        
        self.alpha = alpha
        self.weights_ = None
    
    def fit(self, X: Matrix, y: Matrix) -> 'RidgeRegressionClosedForm':
        """
        Compute optimal weights using closed-form solution.
        
        THE ALGORITHM:
        -------------
        1. Compute XᵀX (Gram matrix)
        2. Add αI to the diagonal (regularization)
        3. Compute (XᵀX + αI)⁻¹ (matrix inversion - expensive!)
        4. Compute Xᵀy
        5. Multiply: w = (XᵀX + αI)⁻¹Xᵀy
        
        BIAS TERM HANDLING:
        ------------------
        To avoid regularizing the bias, we have two options:
        1. Don't add α to the first diagonal element (modify the identity matrix)
        2. Center the data (make it zero-mean) so bias is naturally zero
        
        We use option 1 for simplicity.
        
        PARAMETERS:
        ----------
        X : Matrix
            Training features
        y : Matrix
            Training targets
            
        RETURNS:
        -------
        self : RidgeRegressionClosedForm
        """
        
        if X.num_rows != y.num_rows:
            raise ValueError("X and y must have same number of samples")
        
        n_features = X.num_cols
        n_samples = X.num_rows
        
        # =====================================================================
        # STEP 1: Compute XᵀX
        # =====================================================================
        XtX_elements = []
        for i in range(n_features):
            row = []
            for j in range(n_features):
                col_i = X.column(i)
                col_j = X.column(j)
                element = col_i.dot(col_j)
                row.append(element)
            XtX_elements.append(row)
        
        XtX = Matrix(XtX_elements)
        
        # =====================================================================
        # STEP 2: Add αI (regularization term)
        # =====================================================================
        # Create (XᵀX + αI)
        # Special: Don't add α to first diagonal element (don't regularize bias)
        XtX_plus_alphaI = []
        for i in range(n_features):
            row = []
            for j in range(n_features):
                if i == j:
                    # Diagonal element
                    if i == 0:
                        # Don't regularize bias
                        row.append(XtX[i, j])
                    else:
                        # Add α for other features
                        row.append(XtX[i, j] + self.alpha)
                else:
                    # Off-diagonal element
                    row.append(XtX[i, j])
            XtX_plus_alphaI.append(row)
        
        XtX_reg = Matrix(XtX_plus_alphaI)
        
        # =====================================================================
        # STEP 3: Invert (XᵀX + αI)
        # =====================================================================
        XtX_reg_inv = self._invert_matrix(XtX_reg)
        
        # =====================================================================
        # STEP 4: Compute Xᵀy
        # =====================================================================
        Xty_elements = []
        for j in range(n_features):
            col_j = X.column(j)
            y_vec = Vector([y[i, 0] for i in range(n_samples)])
            element = col_j.dot(y_vec)
            Xty_elements.append(element)
        
        Xty = Vector(Xty_elements)
        
        # =====================================================================
        # STEP 5: Compute w = (XᵀX + αI)⁻¹Xᵀy
        # =====================================================================
        weights_elements = []
        for i in range(n_features):
            row_i = XtX_reg_inv.row(i)
            weight_i = row_i.dot(Xty)
            weights_elements.append(weight_i)
        
        self.weights_ = Vector(weights_elements)
        
        return self
    
    def predict(self, X: Matrix) -> Matrix:
        """Make predictions: ŷ = Xw"""
        
        if self.weights_ is None:
            raise RuntimeError("Model not trained! Call .fit() first.")
        
        predictions = []
        for i in range(X.num_rows):
            x_i = X.row(i)
            y_pred = x_i.dot(self.weights_)
            predictions.append([y_pred])
        
        return Matrix(predictions)
    
    def score(self, X: Matrix, y: Matrix) -> float:
        """Compute R² score."""
        
        y_pred = self.predict(X)
        
        y_values = [y[i, 0] for i in range(y.num_rows)]
        y_mean = sum(y_values) / len(y_values)
        
        ss_residual = sum((y[i, 0] - y_pred[i, 0]) ** 2 for i in range(y.num_rows))
        ss_total = sum((y[i, 0] - y_mean) ** 2 for i in range(y.num_rows))
        
        if ss_total < 1e-10:
            return 1.0 if ss_residual < 1e-10 else 0.0
        
        return 1.0 - (ss_residual / ss_total)
    
    def _invert_matrix(self, M: Matrix) -> Matrix:
        """
        Invert a matrix using Gaussian elimination.
        
        This is the same implementation as in LinearRegressionClosedForm.
        For production code, you'd use a optimized linear algebra library.
        """
        
        n = M.num_rows
        
        if M.num_rows != M.num_cols:
            raise ValueError("Can only invert square matrices")
        
        # Create augmented matrix [M | I]
        aug = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(float(M[i, j]))
            for j in range(n):
                row.append(1.0 if i == j else 0.0)
            aug.append(row)
        
        # Gaussian elimination with partial pivoting
        for col in range(n):
            # Find pivot
            max_row = col
            max_val = abs(aug[col][col])
            
            for row in range(col + 1, n):
                if abs(aug[row][col]) > max_val:
                    max_val = abs(aug[row][col])
                    max_row = row
            
            if abs(aug[max_row][col]) < 1e-10:
                raise ValueError(f"Matrix is singular at column {col}")
            
            # Swap rows
            if max_row != col:
                aug[col], aug[max_row] = aug[max_row], aug[col]
            
            # Scale pivot row
            pivot = aug[col][col]
            for j in range(2 * n):
                aug[col][j] /= pivot
            
            # Eliminate column
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    for j in range(2 * n):
                        aug[row][j] -= factor * aug[col][j]
        
        # Extract inverse from right half
        inv_elements = []
        for i in range(n):
            row = [aug[i][j] for j in range(n, 2 * n)]
            inv_elements.append(row)
        
        return Matrix(inv_elements)

