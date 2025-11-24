"""
THE MATHEMATICAL SETUP:
----------------------
Given data (X, y):
- X is a matrix of features (n_samples × n_features)
- y is a vector of targets (n_samples × 1)

We want to learn weights w such that:
    ŷ = Xw ≈ y

Where ŷ are our predictions.

We measure "how wrong" using Mean Squared Error:
    Loss = (1/n) Σ(ŷᵢ - yᵢ)²

The gradient tells us how to improve:
    ∇Loss = (2/n) Xᵀ(ŷ - y)

We update weights by moving opposite to the gradient:
    w := w - learning_rate × ∇Loss

"""

from typing import Literal

from foundations.gradient_descent.batch_gd import BatchGradientDescent
from foundations.gradient_descent.loss_strategies import MSELoss
from foundations.gradient_descent.mini_batch_gd import MiniBatchGradientDescent
from foundations.gradient_descent.sdg import StochasticGradientDescent
from foundations.linear_algebra.vectors_and_matrices import Matrix


class LinearRegression:
    """
    Linear Regression: Fit a linear relationship between features X and target y.

    MATHEMATICAL MODEL:
    ------------------
    ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = wᵀx
    
    In matrix form: ŷ = Xw
    
    Where:
    - X is the feature matrix (n_samples × n_features)
    - w is the weight vector (n_features × 1)
    - ŷ is the prediction vector (n_samples × 1)
    
    THE BIAS TERM:
    -------------
    The bias (intercept) w₀ allows the line to shift up/down, not just rotate.
    We handle it by adding a column of ones to X:
    
    Original X: [[x₁₁, x₁₂],    With bias: [[1, x₁₁, x₁₂],
                 [x₂₁, x₂₂]]                 [1, x₂₁, x₂₂]]
    
    Now the first weight naturally becomes the bias!
    
    THE OPTIMIZATION:
    ----------------
    We use gradient descent to minimize Mean Squared Error:
        Loss = (1/n) Σ(ŷᵢ - yᵢ)²
    
    The gradient is:
        ∇Loss = (2/n) Xᵀ(ŷ - y)
    
    We update weights iteratively:
        w := w - α·∇Loss
    
    PARAMETERS:
    ----------
    optimizer : str, default='batch'
        Which gradient descent variant to use:
        - 'batch': Use all data per update (smooth but slow)
        - 'sgd': Use one sample per update (fast but noisy)
        - 'mini_batch': Use small batches (balanced)
        
    learning_rate : float, default=0.01
        Step size for gradient descent.
        - Too large: Divergence (loss increases!)
        - Too small: Slow convergence (takes forever)
        - Just right: Steady decrease toward minimum
        
    n_epochs : int, default=100
        Number of passes through the dataset.
        Each epoch processes all training samples.
        
    batch_size : int, default=32
        Only used for mini_batch optimizer.
        Number of samples per batch.
        
    verbose : bool, default=False
        If True, print loss during training.
        Very useful for debugging!
        
    ATTRIBUTES (learned during fit):
    -------------------------------
    weights_ : Vector
        Learned weight vector (including bias as first element).
        
    history_ : dict
        Training history:
        - 'loss': Loss at each epoch
        - 'epoch': Epoch numbers
        
    optimizer_ : GradientDescentOptimizer
        The optimizer instance used for training.
        
    EXAMPLE USAGE:
    -------------
    >>> from foundations.linear_algebra.vectors_and_matrices import Matrix
    >>> 
    >>> # Create simple dataset: y = 2x + 1
    >>> X = Matrix([[1, 1], [1, 2], [1, 3], [1, 4]])  # Bias column included
    >>> y = Matrix([[3], [5], [7], [9]])
    >>> 
    >>> # Train model
    >>> model = LinearRegression(learning_rate=0.01, n_epochs=100)
    >>> model.fit(X, y)
    >>> 
    >>> # Make predictions
    >>> X_test = Matrix([[1, 5]])
    >>> prediction = model.predict(X_test)
    >>> print(prediction[0, 0])  # Should be close to 11
    """
    
    def __init__(
        self, 
        optimizer: Literal['batch','sgd','mini_batch'] = 'batch',
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = False
    ):
        self.optimizer_type = optimizer
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        
        # These will be set during fit()
        self.weights_ = None
        self.optimizer_ = None
        self.history_ = None
        
    def fit(self, X:Matrix, y:Matrix)-> 'LinearRegression':
        """
        Train the linear regression model using gradient descent.
        
        This is where the learning happens! We start with random weights (zeros)
        and iteratively adjust them to minimize the loss.
        
        WHAT HAPPENS DURING TRAINING:
        -----------------------------
        For each epoch:
            1. Compute predictions: ŷ = Xw
            2. Compute loss: L = (1/n) Σ(ŷᵢ - yᵢ)²
            3. Compute gradient: ∇L = (2/n) Xᵀ(ŷ - y)
            4. Update weights: w := w - α·∇L
            5. Check if converged (loss stopped decreasing)
        
        THE OPTIMIZER DOES THE WORK:
        ---------------------------
        Instead of manually implementing gradient descent, I use my optimizer
        classes from foundations/gradient_descent/. This separation of concerns
        is beautiful:
        - LinearRegression defines WHAT to minimize (MSE loss)
        - Optimizer defines HOW to minimize (Batch, SGD, Mini-Batch)
        
        This means I can swap optimizers without changing LinearRegression!
        
        PARAMETERS:
        ----------
        X : Matrix, shape (n_samples, n_features)
            Training data. Should include bias column (column of ones)!
            
        y : Matrix, shape (n_samples, 1)
            Target values (what we're trying to predict).
            
        RETURNS:
        -------
        self : LinearRegression
            Returns self for method chaining (model.fit(X,y).predict(X_test))
            
        IMPORTANT NOTES:
        ---------------
        - X MUST include a bias column (first column of ones)
        - If features have very different scales, consider normalizing first
        - Watch the loss during training (set verbose=True) to check convergence
        - If loss increases, learning rate is too large!
        - If loss decreases too slowly, learning rate might be too small
    
        """
        # Step1 : Validation
        if X.num_rows != y.num_rows:
            raise ValueError(
                f"X has {X.num_rows} samples but y has {y.num_rows} samples. "
                "They must match!"
            )
        
        if y.num_cols != 1:
            raise ValueError(
                f"y must be a column vector (n×1), but got shape {y.shape}. "
                "Use Matrix([[y1], [y2], ...]) format."
            )
        
        # Step2: Create the optimizer
        
        loss_function = MSELoss()
        
        match self.optimizer_type:
            case 'batch':
                self.optimizer_ = BatchGradientDescent(
                    loss_function,
                    learning_rate=self.learning_rate,
                    n_epochs=self.n_epochs,
                    verbose=self.verbose
                )
            case 'sgd':
                self.optimizer_ = StochasticGradientDescent(
                    loss_function=loss_function,
                    learning_rate=self.learning_rate,
                    n_epochs=self.n_epochs,
                    verbose=self.verbose
                )

            case 'mini_batch':
                self.optimizer_ = MiniBatchGradientDescent(
                    loss_function=loss_function,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate,
                    n_epochs=self.n_epochs,
                    verbose=self.verbose
                )

            case _:
                raise ValueError(
                    f"Unknown optimizer '{self.optimizer_type}'. "
                    "Choose from: 'batch', 'sgd', 'mini_batch'")
                
        # Step3: Train the model
        if self.verbose:
            print("=" * 70)
            print("TRAINING LINEAR REGRESSION MODEL")
            print("=" * 70)
            print(f"Optimizer: {self.optimizer_type}")
            print(f"Dataset: {X.num_rows} samples, {X.num_cols} features")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Epochs: {self.n_epochs}")
            print("-" * 70)
        
        # This single line does all the training for us
        self.optimizer_.fit(X,y)
        
        # Extract learned weights from optimizer
        self.weights_ = self.optimizer_.weights_
        self.history_ = self.optimizer_.history_
        
        if self.verbose:
            print("-" * 70)
            print("TRAINING COMPLETE!")
            print(f"Final loss: {self.history_['loss'][-1]:.6f}")
            print(f"Learned weights: {[f'{w:.4f}' for w in self.weights_.elements]}")
            
            # Interpret weights for user
            print("\nWEIGHT INTERPRETATION:")
            print(f"  Bias (w0): {self.weights_[0]:.4f}")
            for i in range(1, len(self.weights_)):
                print(f"  Feature {i} (w{i}): {self.weights_[i]:.4f}")
            
            print("=" * 70)
        
        return self  # Return self for method chaining
    
    def predict(self, X:Matrix) -> Matrix:
        """
        Make predictions using the learned weights.
        
        After training, we use the learned weights to predict y values for new X values.
        This is the simplest part: just compute ŷ = Xw (matrix-vector multiplication).
        
        THE PREDICTION FORMULA:
        ----------------------
        ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
        
        In matrix form: ŷ = Xw
        
        For each sample xᵢ in X:
            ŷᵢ = xᵢᵀw = Σⱼ wⱼxᵢⱼ
        
        PARAMETERS:
        ----------
        X : Matrix, shape (n_samples, n_features)
            New data to make predictions for.
            MUST have same number of features as training data!
            MUST include bias column (first column of ones)!
            
        RETURNS:
        -------
        predictions : Matrix, shape (n_samples, 1)
            Predicted target values for each sample.
            
        INTERPRETATION:
        --------------
        Each prediction is a weighted sum of features:
        - Bias w₀: Shifts predictions up/down
        - Feature weights wᵢ: Control influence of each feature
        - Positive weight: Feature increases → prediction increases
        - Negative weight: Feature increases → prediction decreases
        - Weight magnitude: How much that feature matters
        
        """
        # Check if model has been trained
        if self.weights_ is None:
            raise RuntimeError(
                "Model not trained yet! Call .fit(X, y) before .predict(X)."
            )
        
        # Check dimensions match training data
        if X.num_cols != len(self.weights_):
            raise ValueError(
                f"X has {X.num_cols} features but model was trained on "
                f"{len(self.weights_)} features. "
                "Did you forget to add the bias column?"
            )
        
        # Make predictions: ŷ = Xw
        # For each sample, compute dot product of features and weights
        predictions = []
        for i in range(X.num_rows):
            # Get i-th sample as a vector
            x_i = X.row(i)
            
            # Compute prediction: ŷᵢ = xᵢᵀw
            y_pred = x_i.dot(self.weights_)
            
            predictions.append([y_pred])
        
        return Matrix(predictions)
    
    def score(self, X: Matrix, y: Matrix) -> float:
        """
        Compute R² score (coefficient of determination).
        
        R² measures how well the model fits the data:
        - R² = 1: Perfect fit (model explains all variance)
        - R² = 0: Model no better than predicting the mean
        - R² < 0: Model worse than predicting the mean (very bad!)
        
        THE FORMULA:
        -----------
        R² = 1 - (SS_residual / SS_total)
        
        Where:
        - SS_residual = Σ(yᵢ - ŷᵢ)² (unexplained variance)
        - SS_total = Σ(yᵢ - ȳ)² (total variance)
        
        INTERPRETATION:
        --------------
        R² = 0.8 means: "The model explains 80% of the variance in y"
        
        The remaining 20% is either:
        - Noise in the data
        - Non-linear patterns the linear model can't capture
        - Missing features
        
        PARAMETERS:
        ----------
        X : Matrix
            Features to predict on
        y : Matrix
            True target values
            
        RETURNS:
        -------
        r2 : float
            R² score between -∞ and 1
        """
        # Make predictions
        y_pred = self.predict(X)
        
        # Compute mean of y
        y_values = [y[i, 0] for i in range(y.num_rows)]
        y_mean = sum(y_values) / len(y_values)
        
        # Compute SS_residual (sum of squared errors)
        ss_residual = sum((y[i, 0] - y_pred[i, 0]) ** 2 
                         for i in range(y.num_rows))
        
        # Compute SS_total (total variance)
        ss_total = sum((y[i, 0] - y_mean) ** 2 
                      for i in range(y.num_rows))
        
        # Compute R²
        if ss_total == 0:
            # All y values are the same (no variance to explain)
            return 1.0 if ss_residual == 0 else 0.0
        
        r2 = 1.0 - (ss_residual / ss_total)
        return r2
    
    def get_params(self) -> dict:
        """
        Get model parameters (for compatibility with sklearn-style interfaces).
        
        Returns:
        -------
        params : dict
            Dictionary of model hyperparameters
        """
        return {
            'optimizer': self.optimizer_type,
            'learning_rate': self.learning_rate,
            'n_epochs': self.n_epochs,
            'batch_size': self.batch_size,
            'verbose': self.verbose
        }