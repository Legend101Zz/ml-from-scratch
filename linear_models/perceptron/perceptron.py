"""
THE PERCEPTRON: The First Artificial Neuron (1958)
==================================================

This module implements the perceptron algorithm - the grandfather of all neural
networks. When I first learned about this, I was amazed that something invented
in 1958 is still the fundamental building block of modern deep learning!

Frank Rosenblatt built actual hardware that could learn using this algorithm.
Imagine: a physical machine that learns from examples. In 1958! That was
revolutionary.

MY LEARNING JOURNEY:
-------------------
Coming from logistic regression, the perceptron seemed "too simple." No sigmoid,
no cross-entropy, no careful gradient computation. Just: if wrong, adjust weights.

But implementing it taught me profound lessons:
1. Simple doesn't mean ineffective (for separable data, it's perfect)
2. The step function is both a strength (simple) and weakness (not differentiable)
3. Online learning is natural and elegant
4. Sometimes you don't need probabilities, just decisions

And most importantly: every neuron in a billion-parameter model is fundamentally
doing what the perceptron does!

THE ALGORITHM:
-------------
The perceptron makes binary classifications using a step function:

Step 1: Compute linear combination
    z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = w^T x

Step 2: Apply step function
    ŷ = { 1  if z ≥ 0
        { 0  if z < 0

Step 3: Update weights (only if wrong!)
    If ŷ ≠ y:
        w := w + α(y - ŷ)x

That's it! The perceptron only learns from mistakes.

THE GEOMETRIC INTUITION:
-----------------------
The weight vector w is perpendicular to the decision boundary (the hyperplane
where w^T x = 0).

When we make a mistake:
- If we predicted 0 but should be 1: w := w + αx
  This rotates w toward x, moving the boundary to include this point
  
- If we predicted 1 but should be 0: w := w - αx
  This rotates w away from x, moving the boundary to exclude this point

Each update nudges the boundary in the right direction!

THE CONVERGENCE THEOREM:
-----------------------
If the data is linearly separable (a hyperplane can perfectly separate classes),
the perceptron is GUARANTEED to converge to a perfect solution in finite steps!

This is a mathematical theorem proved by Rosenblatt. No matter what initialization,
no matter the order of examples, if a solution exists, perceptron finds it.

The catch: if data isn't separable, it will never converge. It will cycle through
mistakes forever.

PERCEPTRON VS LOGISTIC REGRESSION:
----------------------------------
Perceptron:
- Step activation: hard decision
- No probabilities (just 0 or 1)
- Updates only on mistakes
- Simpler, faster
- Guaranteed convergence for separable data
- Can't handle non-separable data gracefully

Logistic Regression:
- Sigmoid activation: soft decision
- Gives probabilities (confidence)
- Updates on every example
- More complex, slower
- Converges to best fit even for non-separable data
- Handles overlapping classes gracefully

Author: Mrigesh (appreciating where it all began)
"""


from foundations.activations import step
from foundations.linear_algebra.vectors_and_matrices import Matrix, Vector


class Perceptron:
    """
    The Perceptron: Binary linear classifier using step activation.
    
    This implements the original perceptron algorithm from 1958. It learns a
    linear decision boundary by updating weights only when it makes mistakes.
    
    THE MODEL:
    ---------
    Given features x, the perceptron predicts:
        z = w^T x (linear combination)
        ŷ = step(z) = { 1  if z ≥ 0
                      { 0  if z < 0
    
    THE UPDATE RULE:
    ---------------
    Only update when prediction is wrong:
    
    If ŷ ≠ y:
        w := w + α(y - ŷ)x
    
    Where:
    - α is the learning rate
    - (y - ŷ) is +1 or -1
    - x is the input vector
    
    BREAKING DOWN THE UPDATE:
    ------------------------
    Case 1: Predicted 0, should be 1
        - Error = 1 - 0 = 1
        - Update: w := w + αx
        - Effect: Increase w^T x (move toward 1)
    
    Case 2: Predicted 1, should be 0
        - Error = 0 - 1 = -1
        - Update: w := w - αx
        - Effect: Decrease w^T x (move toward 0)
    
    THE CONVERGENCE GUARANTEE:
    -------------------------
    For linearly separable data, the perceptron will find a perfect separator
    in finite steps. No assumptions about initialization, no dependence on order.
    This is proven mathematically!
    
    But: if data isn't separable, it will never converge. It will keep cycling
    through mistakes indefinitely.
    
    LEARNING MODES:
    --------------
    Online (default): Update after each example
        - Follows original perceptron algorithm
        - Good for streaming data
        - Updates are immediate
    
    Batch: Update after seeing all examples
        - Collect all mistakes in an epoch
        - Update on all at once
        - More stable, less noisy
    
    PARAMETERS:
    ----------
    learning_rate : float, default=1.0
        Step size for weight updates
        For perceptron, α=1.0 works well (updates are already discrete)
        
    n_epochs : int, default=100
        Maximum number of passes through data
        May converge earlier if achieves perfect accuracy
        
    mode : str, default='online'
        'online': Update after each example
        'batch': Update after each epoch
        
    verbose : bool, default=False
        Whether to print training progress
        
    ATTRIBUTES:
    ----------
    weights_ : Vector
        Learned weights (including bias as first element)
        
    history_ : dict
        Training history with 'mistakes' and 'epoch'
        
    converged_ : bool
        Whether algorithm converged (no mistakes in an epoch)
        
    EXAMPLE USAGE:
    -------------
    >>> from foundations.data_preprocessing.feature_engineering import (
    ...     add_bias_column, train_test_split
    ... )
    >>> 
    >>> # Generate linearly separable data
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> 
    >>> # Add bias (perceptron needs it!)
    >>> X_train_bias = add_bias_column(X_train)
    >>> X_test_bias = add_bias_column(X_test)
    >>> 
    >>> # Train perceptron
    >>> perceptron = Perceptron(learning_rate=1.0, n_epochs=100)
    >>> perceptron.fit(X_train_bias, y_train)
    >>> 
    >>> # Make predictions
    >>> predictions = perceptron.predict(X_test_bias)
    >>> 
    >>> # Check if it converged
    >>> if perceptron.converged_:
    ...     print("Perfect separation achieved!")
    """
    
    def __init__(
        self,
        learning_rate: float = 1.0,
        n_epochs: int = 100,
        mode: str = 'online',
        verbose: bool = False
    ):
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        
        if n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {n_epochs}")
        
        if mode not in ['online', 'batch']:
            raise ValueError(f"mode must be 'online' or 'batch', got {mode}")
        
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.mode = mode
        self.verbose = verbose
        
        # Will be set during training
        self.weights_ = None
        self.history_ = None
        self.converged_ = False
    
    def fit(self, X: Matrix, y: Matrix) -> 'Perceptron':
        """
        Train the perceptron using the classic learning algorithm.
        
        THE TRAINING LOOP:
        -----------------
        Initialize weights to zeros
        
        For each epoch:
            mistakes = 0
            For each example (x, y):
                Compute prediction: ŷ = step(w^T x)
                If ŷ ≠ y:
                    w := w + α(y - ŷ)x
                    mistakes += 1
            
            If mistakes == 0:
                Converged! (found perfect separator)
                Break
        
        THE BEAUTY OF SIMPLICITY:
        ------------------------
        Notice what's NOT here:
        - No loss function computation
        - No gradient calculation
        - No sigmoid or other smooth activation
        - No regularization needed
        
        Just: if wrong, fix it. Repeat until right (or give up after n_epochs).
        
        ONLINE VS BATCH:
        ---------------
        Online: Update immediately after each mistake
            - Original perceptron algorithm
            - Weight vector changes frequently
            - Good for streaming data
        
        Batch: Collect all mistakes, update once per epoch
            - More stable (all updates in same direction)
            - Weight vector changes less frequently
            - Better for non-separable data (more robust)
        
        CONVERGENCE:
        -----------
        If the data is linearly separable, this algorithm will converge in finite
        steps - guaranteed! It will achieve 100% training accuracy.
        
        If the data is NOT separable, it will never converge. After n_epochs, it
        will stop, but there will still be mistakes.
        
        PARAMETERS:
        ----------
        X : Matrix, shape (n_samples, n_features)
            Training features (should include bias column!)
            
        y : Matrix, shape (n_samples, 1)
            Training labels (must be 0 or 1)
            
        RETURNS:
        -------
        self : Perceptron
            Returns self for method chaining
        """
        
        if X.num_rows != y.num_rows:
            raise ValueError(
                f"X has {X.num_rows} samples but y has {y.num_rows} samples"
            )
        
        if y.num_cols != 1:
            raise ValueError(f"y must be column vector, got shape {y.shape}")
        
        # Verify binary labels
        y_values = set(y[i, 0] for i in range(y.num_rows))
        if not y_values.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"y must contain only 0 and 1, got values: {y_values}"
            )
        
        n_samples = X.num_rows
        n_features = X.num_cols
        
        # Initialize weights to zeros
        self.weights_ = Vector([0.0] * n_features)
        
        # Training history
        self.history_ = {
            'mistakes': [],
            'epoch': []
        }
        
        if self.verbose:
            print("=" * 70)
            print("TRAINING PERCEPTRON")
            print("=" * 70)
            print(f"Dataset: {n_samples} samples, {n_features} features")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Mode: {self.mode}")
            print(f"Max epochs: {self.n_epochs}")
            print("-" * 70)
        
        # Training loop
        for epoch in range(self.n_epochs):
            mistakes_this_epoch = 0
            
            # Collect mistakes (for batch mode)
            mistake_updates = []
            
            # =================================================================
            # LOOP THROUGH ALL TRAINING EXAMPLES
            # =================================================================
            
            for i in range(n_samples):
                # Get this example
                x_i = X.row(i)
                y_i = y[i, 0]
                
                # ============================================================
                # FORWARD PASS: Make prediction
                # ============================================================
                
                # Compute linear combination
                z = x_i.dot(self.weights_)
                
                # Apply step function
                y_pred = 1.0 if z >= 0 else 0.0
                
                # ============================================================
                # UPDATE: Only if prediction is wrong!
                # ============================================================
                
                if y_pred != y_i:
                    mistakes_this_epoch += 1
                    
                    # Compute update: α(y - ŷ)x
                    error = y_i - y_pred  # Will be +1 or -1
                    
                    if self.mode == 'online':
                        # Update immediately (online learning)
                        updated_weights = []
                        for j in range(n_features):
                            new_weight = (self.weights_[j] + 
                                        self.learning_rate * error * x_i[j])
                            updated_weights.append(new_weight)
                        
                        self.weights_ = Vector(updated_weights)
                    
                    else:  # batch mode
                        # Store for batch update later
                        mistake_updates.append((error, x_i))
            
            # =================================================================
            # BATCH UPDATE (if using batch mode)
            # =================================================================
            
            if self.mode == 'batch' and mistake_updates:
                # Accumulate all updates
                accumulated_update = [0.0] * n_features
                
                for error, x_i in mistake_updates:
                    for j in range(n_features):
                        accumulated_update[j] += self.learning_rate * error * x_i[j]
                
                # Apply accumulated update
                updated_weights = []
                for j in range(n_features):
                    new_weight = self.weights_[j] + accumulated_update[j]
                    updated_weights.append(new_weight)
                
                self.weights_ = Vector(updated_weights)
            
            # =================================================================
            # RECORD HISTORY AND CHECK CONVERGENCE
            # =================================================================
            
            self.history_['mistakes'].append(mistakes_this_epoch)
            self.history_['epoch'].append(epoch)
            
            if self.verbose and (epoch % 10 == 0 or epoch == self.n_epochs - 1):
                print(f"Epoch {epoch:4d}: {mistakes_this_epoch} mistakes")
            
            # Check for convergence (perfect classification!)
            if mistakes_this_epoch == 0:
                self.converged_ = True
                if self.verbose:
                    print(f"\n✓ CONVERGED after {epoch + 1} epochs!")
                    print("  Perfect separation achieved!")
                break
        
        if self.verbose:
            if not self.converged_:
                print("\n✗ Did not converge within maximum epochs")
                print("  Data may not be linearly separable")
            print("-" * 70)
            print(f"Final mistakes: {self.history_['mistakes'][-1]}")
            print(f"Total updates: {sum(self.history_['mistakes'])}")
            print("=" * 70)
        
        return self
    
    def predict(self, X: Matrix) -> Matrix:
        """
        Predict class labels using the learned decision boundary.
        
        THE PREDICTION:
        --------------
        For each sample x:
            z = w^T x (linear combination)
            ŷ = step(z) = { 1  if z ≥ 0
                          { 0  if z < 0
        
        This is a hard decision - no probabilities, no confidence measures.
        You either get 0 or 1, nothing in between.
        
        THE DECISION BOUNDARY:
        ---------------------
        The boundary is where w^T x = 0. This is a hyperplane.
        - Points where w^T x > 0 are classified as 1
        - Points where w^T x < 0 are classified as 0
        - Points exactly on the boundary (w^T x = 0) are classified as 1 by convention
        
        PARAMETERS:
        ----------
        X : Matrix, shape (n_samples, n_features)
            Features to predict on (must match training format)
            
        RETURNS:
        -------
        predictions : Matrix, shape (n_samples, 1)
            Predicted class labels (0 or 1)
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
            z = x_i.dot(self.weights_)
            y_pred = 1.0 if z >= 0 else 0.0
            predictions.append([y_pred])
        
        return Matrix(predictions)
    
    def score(self, X: Matrix, y: Matrix) -> float:
        """
        Compute classification accuracy.
        
        PARAMETERS:
        ----------
        X : Matrix
            Features
            
        y : Matrix
            True labels
            
        RETURNS:
        -------
        accuracy : float
            Fraction of correct predictions
        """
        
        predictions = self.predict(X)
        
        correct = sum(1 for i in range(y.num_rows)
                     if abs(predictions[i, 0] - y[i, 0]) < 0.5)
        
        return correct / y.num_rows
    
    def get_decision_boundary_params(self) -> dict:
        """
        Get parameters defining the decision boundary.
        
        The decision boundary is where w^T x = 0.
        
        For 2D data (two features x₁, x₂ plus bias):
            w₀ + w₁x₁ + w₂x₂ = 0
            x₂ = -(w₀ + w₁x₁) / w₂
        
        RETURNS:
        -------
        params : dict
            Contains:
            - 'weights': Feature weights (without bias)
            - 'bias': Bias term
            - 'slope': Slope of boundary (for 2D case)
            - 'intercept': Intercept of boundary (for 2D case)
        """
        
        if self.weights_ is None:
            raise RuntimeError("Model not trained!")
        
        n_features = len(self.weights_)
        bias = self.weights_[0]
        weights = [self.weights_[i] for i in range(1, n_features)]
        
        params = {
            'weights': weights,
            'bias': bias
        }
        
        # For 2D case, compute slope and intercept
        if len(weights) == 2:
            w1, w2 = weights
            if abs(w2) > 1e-10:
                slope = -w1 / w2
                intercept = -bias / w2
                params['slope'] = slope
                params['intercept'] = intercept
        
        return params
    
    def decision_function(self, X: Matrix) -> Matrix:
        """
        Compute the decision function values (signed distance from boundary).
        
        This returns w^T x for each sample, BEFORE applying the step function.
        
        The value tells you:
        - Positive: On the "class 1" side of boundary
        - Negative: On the "class 0" side of boundary
        - Zero: Exactly on the boundary
        - Magnitude: How far from the boundary (confidence)
        
        Unlike logistic regression which gives probabilities, the perceptron
        gives signed distances. This is less interpretable but still useful
        for understanding model confidence.
        
        PARAMETERS:
        ----------
        X : Matrix
            Features
            
        RETURNS:
        -------
        scores : Matrix
            Decision function values (w^T x for each sample)
        """
        
        if self.weights_ is None:
            raise RuntimeError("Model not trained!")
        
        scores = []
        for i in range(X.num_rows):
            x_i = X.row(i)
            z = x_i.dot(self.weights_)
            scores.append([z])
        
        return Matrix(scores)