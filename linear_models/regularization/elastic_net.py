"""
ELASTIC NET: Combining L1 and L2 Regularization
===============================================

This module implements Elastic Net, which combines Ridge (L2) and Lasso (L1) 
regularization to get the best of both worlds. When I first learned about this, 
I thought "Why not just pick one?" Then I encountered a dataset with correlated 
features and discovered that Elastic Net solved problems that neither Ridge nor 
Lasso could handle alone.

MY LEARNING JOURNEY:
-------------------
I had a dataset predicting wine quality with 15 chemical features. Many were 
correlated (acidity measures, sulfate variants, etc.). 

With Lasso (pure L1):
- It arbitrarily picked one feature from correlated groups
- Results were unstable - small data changes flipped which feature was selected
- Performance was mediocre

With Ridge (pure L2):
- Kept all features with small weights
- No feature selection - model wasn't interpretable
- Performance was decent but not great

With Elastic Net (L1 + L2):
- Selected representative features from correlated groups
- Stable selections across different data splits
- Best test performance!
- Interpretable AND accurate

That's when I understood: Elastic Net isn't just a compromise, it's genuinely 
better than either method alone for many real-world problems.

THE BIG INSIGHT:
---------------
Lasso and Ridge each have weaknesses:

Lasso's weakness: With correlated features (x₁ ≈ x₂), Lasso tends to:
- Arbitrarily pick one and zero the other
- Be unstable (which one gets picked can change with small data variations)
- Struggle to select groups of related features

Ridge's weakness:
- Never eliminates features (no sparsity)
- Doesn't do feature selection
- Less interpretable with many features

Elastic Net combines penalties to get:
- Sparsity from L1 (feature selection)
- Stability from L2 (handles correlated features)
- Best of both worlds!

THE MATHEMATICS:
---------------
Elastic Net minimizes:

Loss = MSE + λ₁ × Σ|wᵢ| + λ₂ × Σwᵢ²

Or equivalently (the form we'll use):

Loss = MSE + λ × [α × Σ|wᵢ| + (1-α) × Σwᵢ²]

Where:
- λ controls overall regularization strength
- α ∈ [0,1] controls the L1/L2 mix:
  - α = 0: Pure Ridge (L2 only)
  - α = 1: Pure Lasso (L1 only)
  - α = 0.5: Equal mix (common choice)

THE GEOMETRIC INTUITION:
------------------------
Remember the constraint region visualizations:
- Lasso: Diamond shape (has corners → sparsity)
- Ridge: Circle shape (smooth → no sparsity)
- Elastic Net: Rounded diamond (corners but smoother → sparse but stable!)

The L2 term "rounds off" the sharp corners of the L1 constraint, making the 
solution less likely to land exactly on an axis but still promoting sparsity.

WHY THIS HELPS WITH CORRELATED FEATURES:
----------------------------------------
Imagine two perfectly correlated features: x₂ = 2×x₁

Lasso behavior:
- Might pick x₁ with weight 5, x₂ with weight 0
- Or x₁ with weight 0, x₂ with weight 2.5
- Unstable! Small noise changes which one wins
- Called the "grouping effect" problem

Ridge behavior:
- Keeps both: x₁ with weight 3.3, x₂ with weight 1.7
- Stable but both survive (no sparsity)

Elastic Net behavior:
- Tends to give similar weights to correlated features
- Might keep both or eliminate both together
- More stable than Lasso, sparser than Ridge
- Called the "grouping effect" - keeps or eliminates groups together

THE OPTIMIZATION ALGORITHM:
--------------------------
Like Lasso, Elastic Net requires coordinate descent because of the L1 term. 
But now each update involves both penalties!

For each weight wⱼ, the update is:

wⱼ = soft_threshold(ρⱼ, α×λ) / (1 + (1-α)×λ)

Where:
- soft_threshold handles the L1 penalty (sparsity)
- Division by (1 + (1-α)×λ) handles the L2 penalty (stability)

The L2 term in the denominator "softens" the thresholding, making it less 
aggressive about setting weights to zero. This stabilizes the solution!

CHOOSING THE PARAMETERS:
-----------------------
You have two parameters to tune:

1. λ (lambda): Overall regularization strength
   - Same role as α in Ridge/Lasso
   - Controls how much regularization overall
   - Use cross-validation to choose

2. α (l1_ratio): Mix between L1 and L2
   - α = 0: Pure Ridge
   - α = 0.5: Balanced (often good default)
   - α = 1: Pure Lasso
   - Can also tune via cross-validation

In practice, many people fix α = 0.5 and only tune λ, which works well.

WHEN TO USE ELASTIC NET:
-----------------------
Use Elastic Net when:
- You have correlated features (very common in real data!)
- You want both feature selection AND stability
- Lasso is giving unstable results
- You want the safest general-purpose regularization

Use pure Lasso when:
- Features are mostly uncorrelated
- Maximum sparsity is critical
- You're confident in feature independence

Use pure Ridge when:
- All features are potentially relevant
- Feature selection isn't needed
- You don't care about sparsity

PRACTICAL ADVICE:
----------------
For most real-world problems, I've found:
- Start with Elastic Net (α = 0.5) - it's robust
- If you need maximum sparsity and features are uncorrelated, try Lasso
- If interpretability doesn't matter, Ridge is fastest

Elastic Net is the "safe default" choice for regularization.

"""

from foundations.linear_algebra.vectors_and_matrices import Matrix, Vector


def soft_threshold(z: float, gamma: float) -> float:
    """
    Soft thresholding operator (same as in Lasso).
    
    This handles the L1 penalty component of Elastic Net.
    """
    if z > gamma:
        return z - gamma
    elif z < -gamma:
        return z + gamma
    else:
        return 0.0


class ElasticNet:
    """
    Elastic Net regression combining L1 and L2 regularization.
    
    This combines the benefits of Ridge (stability with correlated features) and 
    Lasso (sparsity and feature selection) into a single powerful method.
    
    THE OBJECTIVE FUNCTION:
    ----------------------
    Elastic Net minimizes:
    
    J(w) = (1/n) Σ(ŷᵢ - yᵢ)² + λ × [α × Σ|wⱼ| + (1-α)/2 × Σwⱼ²]
    
    Where:
    - First term: Mean squared error (fit the data)
    - Second term: Combined penalty
      - α × Σ|wⱼ|: L1 penalty (sparsity, feature selection)
      - (1-α)/2 × Σwⱼ²: L2 penalty (stability, handles correlation)
    - λ: Overall regularization strength
    - α: L1/L2 mixing parameter (0 to 1)
    
    THE COORDINATE DESCENT UPDATE:
    -----------------------------
    For each weight wⱼ, we optimize it while holding others fixed.
    
    The update rule combines both penalties:
    
    wⱼ = soft_threshold(ρⱼ, λ×α) / (1 + λ×(1-α))
    
    Breaking this down:
    1. ρⱼ is the gradient direction (correlation with residuals)
    2. soft_threshold(ρⱼ, λ×α) handles the L1 penalty
       - Creates sparsity
       - Sets small weights to zero
    3. Division by (1 + λ×(1-α)) handles the L2 penalty
       - Shrinks all weights proportionally
       - Stabilizes the solution
    
    THE BEAUTY OF THIS FORMULA:
    --------------------------
    Notice how the two penalties work together:
    
    - If α = 1 (pure Lasso): denominator is 1, only soft threshold remains
    - If α = 0 (pure Ridge): no soft threshold, only shrinkage in denominator
    - If 0 < α < 1: Both effects combine!
    
    The L2 term in the denominator "softens" the L1 thresholding, making it less 
    aggressive. This is why Elastic Net is more stable than Lasso.
    
    THE GROUPING EFFECT:
    -------------------
    One of Elastic Net's key properties is the "grouping effect": strongly 
    correlated features tend to be selected or eliminated together.
    
    Mathematically, this happens because:
    1. The L2 penalty encourages similar weights for correlated features
    2. The L1 penalty then shrinks the whole group together
    3. Either the whole group survives or gets eliminated
    
    This is much more stable than Lasso's arbitrary selection!
    
    PARAMETERS:
    ----------
    alpha : float, default=1.0
        Overall regularization strength (λ in the formula above)
        Larger values = more regularization
        
    l1_ratio : float, default=0.5
        The α parameter controlling L1/L2 mix
        - l1_ratio=0: Pure Ridge (L2 only)
        - l1_ratio=0.5: Balanced mix (recommended default)
        - l1_ratio=1: Pure Lasso (L1 only)
        
    n_epochs : int, default=100
        Number of passes through all features
        
    tol : float, default=1e-4
        Convergence tolerance (stop if max weight change < tol)
        
    verbose : bool, default=False
        Whether to print training progress
        
    ATTRIBUTES:
    ----------
    weights_ : Vector
        Learned weights (some may be exactly zero)
        
    history_ : dict
        Training history including loss and sparsity
        
    EXAMPLE USAGE:
    -------------
    >>> from foundations.data_preprocessing.feature_engineering import (
    ...     Standardizer, add_bias_column, train_test_split
    ... )
    >>> 
    >>> # Prepare data (standardization is critical!)
    >>> scaler = Standardizer()
    >>> X_train_scaled = scaler.fit_transform(X_train)
    >>> X_test_scaled = scaler.transform(X_test)
    >>> 
    >>> X_train_bias = add_bias_column(X_train_scaled)
    >>> X_test_bias = add_bias_column(X_test_scaled)
    >>> 
    >>> # Train Elastic Net with balanced L1/L2 mix
    >>> model = ElasticNet(alpha=0.1, l1_ratio=0.5, n_epochs=500)
    >>> model.fit(X_train_bias, y_train)
    >>> 
    >>> # Evaluate
    >>> print(f"Test R²: {model.score(X_test_bias, y_test):.4f}")
    >>> print(f"Active features: {len(model.get_active_features())}")
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        n_epochs: int = 100,
        tol: float = 1e-4,
        verbose: bool = False
    ):
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        
        if not 0 <= l1_ratio <= 1:
            raise ValueError(f"l1_ratio must be in [0,1], got {l1_ratio}")
        
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_epochs = n_epochs
        self.tol = tol
        self.verbose = verbose
        
        self.weights_ = None
        self.history_ = None
    
    def fit(self, X: Matrix, y: Matrix) -> 'ElasticNet':
        """
        Train Elastic Net using coordinate descent.
        
        THE ALGORITHM:
        -------------
        Initialize weights to zeros
        
        For each epoch:
            For each feature j:
                # Compute residuals without feature j
                r = y - Σᵢ≠ⱼ wᵢxᵢ
                
                # Compute correlation with residuals
                ρⱼ = xⱼᵀr / ||xⱼ||²
                
                # Apply Elastic Net update (combines L1 and L2)
                wⱼ = soft_threshold(ρⱼ, λ×α) / (1 + λ×(1-α))
                
            Check convergence
        
        THE COORDINATE DESCENT CONVERGENCE:
        ----------------------------------
        Elastic Net is convex, so coordinate descent is guaranteed to converge 
        to the global optimum. Usually converges in 50-300 epochs.
        
        The L2 term actually helps convergence because it makes the problem more 
        strongly convex (better conditioned), so Elastic Net often converges 
        faster than pure Lasso!
        
        PARAMETERS:
        ----------
        X : Matrix, shape (n_samples, n_features)
            Training features (must be standardized and include bias)
            
        y : Matrix, shape (n_samples, 1)
            Training targets
            
        RETURNS:
        -------
        self : ElasticNet
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
            'sparsity': [],
            'epoch': []
        }
        
        if self.verbose:
            print("=" * 70)
            print("TRAINING ELASTIC NET REGRESSION")
            print("=" * 70)
            print(f"Overall strength (λ): {self.alpha}")
            print(f"L1/L2 ratio (α): {self.l1_ratio}")
            print(f"  (α=0: pure Ridge, α=0.5: balanced, α=1: pure Lasso)")
            print(f"Dataset: {n_samples} samples, {n_features} features")
            print(f"Max epochs: {self.n_epochs}")
            print("-" * 70)
        
        # Precompute feature norms
        feature_norms = []
        for j in range(n_features):
            col_j = X.column(j)
            norm_j = sum(val ** 2 for val in col_j.elements)
            feature_norms.append(norm_j)
        
        # Training loop
        for epoch in range(self.n_epochs):
            weights_old = [w for w in self.weights_.elements]
            
            # Coordinate descent: update each weight
            for j in range(n_features):
                # ============================================================
                # Compute residuals without feature j's contribution
                # ============================================================
                residuals = []
                for i in range(n_samples):
                    prediction = sum(self.weights_[k] * X[i, k] 
                                   for k in range(n_features))
                    residual = y[i, 0] - (prediction - self.weights_[j] * X[i, j])
                    residuals.append(residual)
                
                # ============================================================
                # Compute correlation with feature j
                # ============================================================
                col_j = X.column(j)
                rho_j = sum(col_j[i] * residuals[i] for i in range(n_samples))
                
                # ============================================================
                # Apply Elastic Net update
                # ============================================================
                if j == 0:
                    # Don't regularize bias
                    if feature_norms[j] > 1e-10:
                        self.weights_.elements[j] = rho_j / feature_norms[j]
                    else:
                        self.weights_.elements[j] = 0.0
                else:
                    # Elastic Net update: combines L1 and L2
                    if feature_norms[j] > 1e-10:
                        # Normalize by feature norm
                        z = rho_j / feature_norms[j]
                        
                        # L1 penalty strength
                        l1_penalty = self.alpha * self.l1_ratio
                        
                        # L2 penalty strength (in denominator)
                        l2_penalty = self.alpha * (1 - self.l1_ratio)
                        
                        # Soft threshold (L1 part)
                        numerator = soft_threshold(z, l1_penalty)
                        
                        # L2 shrinkage (denominator)
                        denominator = 1.0 + l2_penalty
                        
                        # Combined update
                        self.weights_.elements[j] = numerator / denominator
                    else:
                        self.weights_.elements[j] = 0.0
            
            # ================================================================
            # Check convergence
            # ================================================================
            max_change = max(abs(self.weights_[j] - weights_old[j]) 
                           for j in range(n_features))
            
            # Compute loss
            predictions = []
            for i in range(n_samples):
                pred = sum(self.weights_[j] * X[i, j] for j in range(n_features))
                predictions.append(pred)
            
            mse = sum((predictions[i] - y[i, 0]) ** 2 
                     for i in range(n_samples)) / n_samples
            
            # L1 penalty (don't count bias)
            l1_penalty = sum(abs(self.weights_[j]) for j in range(1, n_features))
            
            # L2 penalty (don't count bias)
            l2_penalty = sum(self.weights_[j] ** 2 for j in range(1, n_features))
            
            # Total loss with both penalties
            loss = (mse + 
                   self.alpha * self.l1_ratio * l1_penalty + 
                   0.5 * self.alpha * (1 - self.l1_ratio) * l2_penalty)
            
            # Count non-zero weights
            n_nonzero = sum(1 for j in range(n_features) 
                          if abs(self.weights_[j]) > 1e-6)
            
            # Record history
            self.history_['loss'].append(loss)
            self.history_['sparsity'].append(n_nonzero)
            self.history_['epoch'].append(epoch)
            
            # Print progress
            if self.verbose and (epoch % 50 == 0 or epoch == self.n_epochs - 1):
                print(f"Epoch {epoch:4d}: Loss = {loss:.6f}, "
                      f"Active = {n_nonzero}/{n_features}, "
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
            print("=" * 70)
        
        return self
    
    def predict(self, X: Matrix) -> Matrix:
        """Make predictions using learned weights."""
        
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
        
        ss_residual = sum((y[i, 0] - y_pred[i, 0]) ** 2 
                         for i in range(y.num_rows))
        ss_total = sum((y[i, 0] - y_mean) ** 2 
                      for i in range(y.num_rows))
        
        if ss_total < 1e-10:
            return 1.0 if ss_residual < 1e-10 else 0.0
        
        return 1.0 - (ss_residual / ss_total)
    
    def get_active_features(self, threshold: float = 1e-6) -> list:
        """
        Get indices of features with non-zero weights.
        
        This tells you which features survived the combined L1+L2 penalty.
        
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