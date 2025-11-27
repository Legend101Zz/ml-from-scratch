"""
LASSO REGRESSION: L1 Regularization and Automatic Feature Selection
===================================================================

This module implements Lasso Regression, which uses L1 regularization to not only
prevent overfitting but also perform automatic feature selection! When I first learned
about this, I thought "Wait, the optimization algorithm can eliminate features
automatically? That's magic!" Then I implemented it and as always discovered it's not magic—
it's beautiful mathematics.

MY LEARNING JOURNEY:
-------------------
I had a dataset with 50 features predicting customer churn. I suspected many features
were useless noise, but which ones? Manual feature selection would be tedious and
arbitrary.

Then I tried Lasso with α=0.1:
- 35 features got weights of EXACTLY ZERO (eliminated!)
- 15 features survived with non-zero weights
- Test accuracy actually IMPROVED compared to using all features
- The model was faster, simpler, and more interpretable

That's when I understood: Lasso doesn't just shrink weights like Ridge—it performs
feature selection as part of the optimization!

THE BIG INSIGHT:
---------------
Ridge (L2) says: "I'll make weights small."
Lasso (L1) says: "I'll make weights small OR ZERO."

This difference seems subtle but has profound implications! Lasso creates sparse
models where many weights are exactly zero. This is automatic feature selection—
the algorithm decides which features to keep and which to discard.

THE MATHEMATICS:
---------------
Lasso minimizes: Loss = (1/n) Σ(ŷᵢ - yᵢ)² + α Σ|wᵢ|
                      = MSE + α × (|w₁| + |w₂| + ... + |wₙ|)

Compare to Ridge: Loss = MSE + α × (w₁² + w₂² + ... + wₙ²)

The difference is subtle (absolute value vs squared) but creates fundamentally
different behavior!

WHY L1 CREATES SPARSITY:
-----------------------
The geometric intuition is beautiful. Imagine 2D weight space (w₁, w₂):

Ridge constraint: w₁² + w₂² ≤ C (circle)
- Smooth, no corners
- Solution rarely lands exactly on axis
- Both weights non-zero (just small)

Lasso constraint: |w₁| + |w₂| ≤ C (diamond)
- Has corners at the axes
- Solution often lands at a corner
- One weight becomes exactly zero!

In high dimensions, Lasso's constraint region has many corners pointing at the axes.
The optimization naturally pushes weights to these corners, setting them to zero.

This isn't a numerical accident—it's the fundamental geometry of the L1 norm!

THE GRADIENT CHALLENGE:
----------------------
Unlike Ridge, we can't use simple gradient descent for Lasso. Why? The absolute
value function isn't differentiable at zero!

The derivative of |w| is:
- +1 when w > 0
- -1 when w < 0
- undefined when w = 0

This creates problems for gradient-based optimization. We need special algorithms:

1. Coordinate Descent: Optimize one weight at a time
2. Proximal Gradient Descent: Use "soft thresholding"
3. Subgradient Methods: Use subgradients instead of gradients

We implement coordinate descent because it's intuitive and works well for Lasso.

COORDINATE DESCENT FOR LASSO:
-----------------------------
Instead of updating all weights simultaneously, we:
1. Fix all weights except wⱼ
2. Find the optimal wⱼ (has a closed-form solution!)
3. Move to next weight
4. Repeat until convergence

For each weight, the optimal update is:
wⱼ = soft_threshold(ρⱼ, α)

Where soft_threshold is:
- ρⱼ - α  if ρⱼ > α
- 0       if |ρⱼ| ≤ α
- ρⱼ + α  if ρⱼ < -α

This "soft thresholding" is the key! It shrinks weights toward zero and sets them
to exactly zero when they're small enough. This is how Lasso eliminates features!

LASSO VS RIDGE: WHEN TO USE WHICH?
----------------------------------
Use Ridge when:
- All features might be relevant
- Features are correlated (Ridge handles this well)
- You want a stable solution
- Interpretability isn't critical

Use Lasso when:
- Many features are likely irrelevant
- You want automatic feature selection
- You need a sparse, interpretable model
- You want to identify the most important features

Use Elastic Net when:
- You want both benefits (feature selection + stability)
- You have correlated features but still want sparsity

THE CHALLENGE WITH CORRELATED FEATURES:
--------------------------------------
Lasso has a quirk: when features are highly correlated, it tends to arbitrarily
pick one and zero the others. This can be unstable—small data changes might swap
which feature gets selected.

Example: If sqft and sqft_meters are perfectly correlated, Lasso might pick one
randomly and zero the other. Ridge would keep both with small weights.

For correlated features, Elastic Net (combining L1 and L2) is more stable.

IMPORTANT: STANDARDIZE FIRST!
-----------------------------
Like Ridge, Lasso penalizes weights based on magnitude. If features have different
scales, this is unfair! Always standardize (mean=0, std=1) before using Lasso.

Also, don't regularize the bias term—only the feature weights.

"""




from foundations.linear_algebra.vectors_and_matrices import Matrix, Vector


def soft_threshold(z: float, gamma: float) -> float:
    """
    Soft thresholding operator (proximal operator for L1 norm).
    
    This is the KEY operation in Lasso! It simultaneously shrinks and sparsifies.
    
    THE FORMULA:
    -----------
    soft_threshold(z, γ) = sign(z) × max(|z| - γ, 0)
    
    Which breaks down to:
    - If z > γ: return z - γ (shrink from above)
    - If z < -γ: return z + γ (shrink from below)
    - If |z| ≤ γ: return 0 (set to zero!)
    
    THE INTUITION:
    -------------
    Imagine z is the "natural" value a weight wants to be (without regularization).
    The soft threshold says: "I'll let you be close to z, but I'm going to:
    1. Push you toward zero by amount γ
    2. If you're too close to zero (|z| ≤ γ), just make you exactly zero"
    
    This is how Lasso eliminates features! Small weights get pushed all the way to zero.
    
    VISUAL EXAMPLE:
    --------------
    Let γ = 1.0
    
    z = 3.0 → soft_threshold(3.0, 1.0) = 2.0 (shrunk but survived)
    z = 0.5 → soft_threshold(0.5, 1.0) = 0.0 (eliminated!)
    z = -2.0 → soft_threshold(-2.0, 1.0) = -1.0 (shrunk but survived)
    z = 0.8 → soft_threshold(0.8, 1.0) = 0.0 (eliminated!)
    
    Notice: Weights with |z| ≤ γ become exactly zero. This is feature selection!
    
    PARAMETERS:
    ----------
    z : float
        The value to threshold
        
    gamma : float
        Threshold parameter (typically related to regularization strength)
        
    RETURNS:
    -------
    result : float
        Soft-thresholded value
    """
    
    if z > gamma:
        return z - gamma
    elif z < -gamma:
        return z + gamma
    else:
        return 0.0


class LassoRegression:
    """
    Lasso Regression with L1 regularization using coordinate descent.
    
    Lasso adds an L1 penalty to linear regression, which not only prevents overfitting
    but also performs automatic feature selection by setting some weights to exactly zero.
    
    THE OBJECTIVE FUNCTION:
    ----------------------
    Lasso minimizes: J(w) = (1/n) Σ(ŷᵢ - yᵢ)² + α Σ|wⱼ|
    
    Where:
    - First term: Mean squared error (fit the data)
    - Second term: L1 penalty (promote sparsity)
    - α: Regularization strength
    
    THE COORDINATE DESCENT ALGORITHM:
    --------------------------------
    We can't use standard gradient descent because |w| isn't differentiable at zero.
    Instead, we use coordinate descent:
    
    For each epoch:
        For each weight wⱼ:
            1. Compute the "partial residual" (removing effect of wⱼ)
            2. Compute the optimal update for wⱼ alone
            3. Apply soft thresholding: wⱼ = soft_threshold(ρⱼ, α)
            4. Update wⱼ
    
    This converges to the optimal solution!
    
    WHY COORDINATE DESCENT WORKS:
    ----------------------------
    The key insight: When we fix all weights except wⱼ, the Lasso problem for wⱼ
    has a simple closed-form solution involving soft thresholding.
    
    By cycling through weights and optimally updating each one, we eventually reach
    the global optimum (Lasso is convex, so coordinate descent is guaranteed to work).
    
    THE SOFT THRESHOLDING STEP:
    --------------------------
    For each weight, the update rule is:
    
    wⱼ = soft_threshold(ρⱼ, α × n)
    
    Where ρⱼ is the correlation between feature j and the current residuals.
    
    The soft threshold:
    - Shrinks ρⱼ toward zero by α
    - Sets wⱼ to exactly zero if |ρⱼ| is too small
    - This is how features get eliminated!
    
    SPARSITY AND FEATURE SELECTION:
    ------------------------------
    As α increases:
    - More weights become exactly zero
    - Fewer features remain "active"
    - The model becomes sparser and more interpretable
    
    α = 0: All features active (standard linear regression)
    α small: Most features active, slightly shrunk
    α medium: Some features eliminated (sweet spot!)
    α large: Most/all features eliminated (predicts mean)
    
    PARAMETERS:
    ----------
    alpha : float, default=1.0
        Regularization strength. Larger values create sparser models.
        
    learning_rate : float, default=0.01
        Not used in pure coordinate descent, but kept for API consistency
        
    n_epochs : int, default=100
        Number of complete passes through all weights
        
    tol : float, default=1e-4
        Convergence tolerance. Stop if max weight change < tol
        
    verbose : bool, default=False
        Whether to print training progress
        
    ATTRIBUTES:
    ----------
    weights_ : Vector
        Learned weights (including bias)
        Many will be exactly zero!
        
    history_ : dict
        Training history including sparsity level
        
    EXAMPLE USAGE:
    -------------
    >>> from foundations.data_preprocessing.feature_engineering import (
    ...     Standardizer, add_bias_column, train_test_split
    ... )
    >>> 
    >>> # Prepare data (standardization is CRITICAL!)
    >>> scaler = Standardizer()
    >>> X_train_scaled = scaler.fit_transform(X_train)
    >>> X_test_scaled = scaler.transform(X_test)
    >>> 
    >>> X_train_bias = add_bias_column(X_train_scaled)
    >>> X_test_bias = add_bias_column(X_test_scaled)
    >>> 
    >>> # Train Lasso
    >>> model = LassoRegression(alpha=0.1, n_epochs=500, verbose=True)
    >>> model.fit(X_train_bias, y_train)
    >>> 
    >>> # Check which features survived
    >>> n_active = sum(1 for w in model.weights_.elements if abs(w) > 1e-6)
    >>> print(f"Active features: {n_active}/{len(model.weights_)}")
    >>> 
    >>> # See the sparse weights
    >>> for i, w in enumerate(model.weights_.elements):
    ...     if abs(w) > 1e-6:
    ...         print(f"Feature {i}: {w:.4f}")
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        learning_rate: float = 0.01,  # Not used, kept for API consistency
        n_epochs: int = 100,
        tol: float = 1e-4,
        verbose: bool = False
    ):
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.tol = tol
        self.verbose = verbose
        
        self.weights_ = None
        self.history_ = None
    
    def fit(self, X: Matrix, y: Matrix) -> 'LassoRegression':
        """
        Train Lasso regression using coordinate descent.
        
        THE COORDINATE DESCENT ALGORITHM:
        --------------------------------
        Initialize weights to zeros
        
        For each epoch:
            For each feature j:
                # Compute residuals WITHOUT feature j's contribution
                r = y - Σᵢ≠ⱼ wᵢxᵢ
                
                # Compute correlation of feature j with residuals
                ρⱼ = xⱼᵀr
                
                # Soft threshold (this is where sparsity happens!)
                wⱼ = soft_threshold(ρⱼ, α × n)
                
            Check convergence (if weights barely changed, stop)
        
        THE MAGIC OF SOFT THRESHOLDING:
        ------------------------------
        When we compute ρⱼ (correlation of feature j with residuals), we're asking:
        "How much does feature j help explain what's left unexplained?"
        
        Soft thresholding says:
        - If |ρⱼ| > α: Keep this feature, but shrink its weight
        - If |ρⱼ| ≤ α: This feature isn't worth the regularization penalty, zero it!
        
        This is how Lasso automatically selects features!
        
        CONVERGENCE:
        -----------
        Coordinate descent is guaranteed to converge for Lasso (the problem is convex).
        We stop when the maximum weight change is below the tolerance.
        
        Typically converges in 50-200 epochs for most problems.
        
        PARAMETERS:
        ----------
        X : Matrix, shape (n_samples, n_features)
            Training features (MUST be standardized and include bias!)
            
        y : Matrix, shape (n_samples, 1)
            Training targets
            
        RETURNS:
        -------
        self : LassoRegression
        """
        
        if X.num_rows != y.num_rows:
            raise ValueError("X and y must have same number of samples")
        
        if y.num_cols != 1:
            raise ValueError("y must be column vector")
        
        n_samples = X.num_rows
        n_features = X.num_cols
        
        # Initialize weights to zeros
        self.weights_ = Vector([0.0] * n_features)
        
        # Training history
        self.history_ = {
            'loss': [],
            'sparsity': [],  # Number of non-zero weights
            'epoch': []
        }
        
        if self.verbose:
            print("=" * 70)
            print("TRAINING LASSO REGRESSION (Coordinate Descent)")
            print("=" * 70)
            print(f"Regularization strength (α): {self.alpha}")
            print(f"Dataset: {n_samples} samples, {n_features} features")
            print(f"Max epochs: {self.n_epochs}")
            print(f"Convergence tolerance: {self.tol}")
            print("-" * 70)
        
        # Precompute feature norms (for normalization)
        # This is needed for the coordinate descent update
        feature_norms = []
        for j in range(n_features):
            col_j = X.column(j)
            norm_j = sum(val ** 2 for val in col_j.elements)
            feature_norms.append(norm_j)
        
        # Training loop
        for epoch in range(self.n_epochs):
            weights_old = [w for w in self.weights_.elements]
            
            # Cycle through each feature (coordinate descent)
            for j in range(n_features):
                # ============================================================
                # STEP 1: Compute residuals WITHOUT feature j
                # ============================================================
                # We need: r = y - Σᵢ≠ⱼ wᵢxᵢ = y - Xw + wⱼxⱼ
                
                residuals = []
                for i in range(n_samples):
                    # Full prediction: Σₖ wₖxᵢₖ
                    prediction = sum(self.weights_[k] * X[i, k] for k in range(n_features))
                    
                    # Remove feature j's contribution: prediction - wⱼxᵢⱼ
                    residual = y[i, 0] - (prediction - self.weights_[j] * X[i, j])
                    residuals.append(residual)
                
                # ============================================================
                # STEP 2: Compute correlation with feature j
                # ============================================================
                # ρⱼ = xⱼᵀr (how much does feature j correlate with residuals?)
                col_j = X.column(j)
                rho_j = sum(col_j[i] * residuals[i] for i in range(n_samples))
                
                # ============================================================
                # STEP 3: Soft threshold (the key step!)
                # ============================================================
                # For bias term (j=0), don't regularize
                if j == 0:
                    # No regularization for bias
                    if feature_norms[j] > 1e-10:
                        self.weights_.elements[j] = rho_j / feature_norms[j]
                    else:
                        self.weights_.elements[j] = 0.0
                else:
                    # Apply soft thresholding for other features
                    if feature_norms[j] > 1e-10:
                        # Normalize by feature norm
                        z = rho_j / feature_norms[j]
                        
                        # Soft threshold with α
                        self.weights_.elements[j] = soft_threshold(z, self.alpha)
                    else:
                        self.weights_.elements[j] = 0.0
            
            # ================================================================
            # Check convergence
            # ================================================================
            max_change = max(abs(self.weights_[j] - weights_old[j]) 
                           for j in range(n_features))
            
            # Compute loss for monitoring
            predictions = []
            for i in range(n_samples):
                pred = sum(self.weights_[j] * X[i, j] for j in range(n_features))
                predictions.append(pred)
            
            mse = sum((predictions[i] - y[i, 0]) ** 2 for i in range(n_samples)) / n_samples
            
            # L1 penalty (don't count bias)
            l1_penalty = sum(abs(self.weights_[j]) for j in range(1, n_features))
            loss = mse + self.alpha * l1_penalty
            
            # Count non-zero weights (sparsity)
            n_nonzero = sum(1 for j in range(n_features) if abs(self.weights_[j]) > 1e-6)
            
            # Record history
            self.history_['loss'].append(loss)
            self.history_['sparsity'].append(n_nonzero)
            self.history_['epoch'].append(epoch)
            
            # Print progress
            if self.verbose and (epoch % 50 == 0 or epoch == self.n_epochs - 1):
                print(f"Epoch {epoch:4d}: Loss = {loss:.6f}, "
                      f"Active features = {n_nonzero}/{n_features}, "
                      f"Max Δw = {max_change:.6f}")
            
            # Check convergence
            if max_change < self.tol:
                if self.verbose:
                    print(f"\nConverged after {epoch + 1} epochs!")
                break
        
        if self.verbose:
            print("-" * 70)
            print("TRAINING COMPLETE!")
            print(f"Final loss: {self.history_['loss'][-1]:.6f}")
            print(f"Active features: {self.history_['sparsity'][-1]}/{n_features}")
            
            # Show which features survived
            print("\nNon-zero weights:")
            for j in range(n_features):
                if abs(self.weights_[j]) > 1e-6:
                    print(f"  Feature {j}: {self.weights_[j]:.6f}")
            
            print("=" * 70)
        
        return self
    
    def predict(self, X: Matrix) -> Matrix:
        """
        Make predictions using learned sparse weights.
        
        Even though many weights are zero, prediction is still just ŷ = Xw.
        The zero weights mean those features are ignored!
        
        PARAMETERS:
        ----------
        X : Matrix
            Features to predict on
            
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
        """Compute R² score."""
        
        y_pred = self.predict(X)
        
        y_values = [y[i, 0] for i in range(y.num_rows)]
        y_mean = sum(y_values) / len(y_values)
        
        ss_residual = sum((y[i, 0] - y_pred[i, 0]) ** 2 for i in range(y.num_rows))
        ss_total = sum((y[i, 0] - y_mean) ** 2 for i in range(y.num_rows))
        
        if ss_total < 1e-10:
            return 1.0 if ss_residual < 1e-10 else 0.0
        
        return 1.0 - (ss_residual / ss_total)
    
    def get_active_features(self, threshold: float = 1e-6) -> list:
        """
        Get indices of features with non-zero weights.
        
        This tells you which features the model selected!
        
        PARAMETERS:
        ----------
        threshold : float
            Values below this are considered zero
            
        RETURNS:
        -------
        active_indices : list
            Indices of non-zero weights
        """
        
        if self.weights_ is None:
            raise RuntimeError("Model not trained!")
        
        active = []
        for i, w in enumerate(self.weights_.elements):
            if abs(w) > threshold:
                active.append(i)
        
        return active
