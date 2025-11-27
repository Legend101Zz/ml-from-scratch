"""
LOGISTIC REGRESSION: The Foundation of Binary Classification
============================================================

This module implements logistic regression from scratch, using gradient descent
to learn a linear decision boundary for binary classification. This was my first
real classification algorithm, and understanding it deeply unlocked all of modern
machine learning classification.

MY LEARNING JOURNEY:
-------------------
Coming from linear regression, I initially tried to use it for classification by
just rounding the output. Predict > 0.5 → class 1, otherwise class 0. This seemed
logical but failed spectacularly!

The problems were immediate:
- Predictions could be negative or greater than 1 (not valid probabilities)
- Extremely sensitive to outliers
- No meaningful probability interpretation
- MSE loss didn't make sense for binary outcomes

Then I discovered logistic regression and everything clicked. The key insight:
don't predict the class directly, predict the PROBABILITY of the class, then
threshold that probability to make a decision.

THE FUNDAMENTAL IDEA:
--------------------
Logistic regression has two steps:

Step 1: Compute a linear combination (just like linear regression)
    z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = w^T x

Step 2: Transform z into a probability using the sigmoid function
    P(y=1|x) = σ(z) = 1 / (1 + e^(-z))

The sigmoid function is the key! It takes any real number and squashes it into
the range (0, 1), giving us a valid probability.

Properties of sigmoid:
- z → -∞: σ(z) → 0 (very confident class is 0)
- z = 0: σ(z) = 0.5 (uncertain, 50-50)
- z → +∞: σ(z) → 1 (very confident class is 1)

THE DECISION BOUNDARY:
---------------------
We predict class 1 when P(y=1|x) ≥ 0.5, which happens when:
    σ(w^T x) ≥ 0.5
    ⟺ w^T x ≥ 0

So the decision boundary is where w^T x = 0. This is a hyperplane in feature space!

For 2D data (two features x₁, x₂):
    w₀ + w₁x₁ + w₂x₂ = 0
    x₂ = -(w₀ + w₁x₁) / w₂

This is a straight line. Points above the line get classified as one class, points
below as the other class.

The weight vector w = [w₁, w₂, ...] is perpendicular to the decision boundary and
points toward the class 1 region. The magnitude of w controls how quickly the
probability changes as you move away from the boundary.

THE LOSS FUNCTION: BINARY CROSS-ENTROPY
---------------------------------------
We use cross-entropy (not MSE!) because it's designed for probability predictions:

    Loss = -(1/n) Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

Where ŷᵢ = σ(w^T xᵢ) is the predicted probability.

Breaking this down by cases:
- If yᵢ = 1: Loss = -log(ŷᵢ)
  - ŷᵢ = 0.9 (confident correct): Loss = 0.105 (low)
  - ŷᵢ = 0.1 (confident wrong): Loss = 2.303 (high)

- If yᵢ = 0: Loss = -log(1-ŷᵢ)
  - ŷᵢ = 0.1 (confident correct): Loss = 0.105 (low)
  - ŷᵢ = 0.9 (confident wrong): Loss = 2.303 (high)

The logarithm creates an exponentially growing penalty for confident mistakes!

WHY NOT MSE?
-----------
I tried MSE initially: Loss = (1/n) Σ(ŷᵢ - yᵢ)²

The problem: the gradient involves σ'(z) = σ(1-σ), which approaches zero when
the model is very confident (even if wrong!). This causes vanishing gradients
and the model gets stuck.

With cross-entropy, the gradient simplifies beautifully to:
    ∂Loss/∂w = (1/n) X^T (ŷ - y)

The sigmoid derivative cancels out! This is identical to linear regression's
gradient, just with sigmoid predictions instead of linear predictions. Training
is fast and stable.

THE GRADIENT DESCENT UPDATE:
---------------------------
For each epoch:
1. Forward pass: compute z = Xw, then ŷ = σ(z)
2. Compute loss: binary cross-entropy
3. Compute gradient: (1/n) X^T (ŷ - y)
4. Update weights: w := w - α × gradient

This iteratively moves w in the direction that reduces the loss.

REGULARIZATION:
--------------
Just like linear regression, logistic regression can overfit with many features.
We add L2 regularization (same as Ridge):

    Loss = Binary Cross-Entropy + λ Σwⱼ²

This penalizes large weights, forcing the model to find simpler solutions that
generalize better. The regularization strength λ controls the tradeoff.

Author: Mrigesh (discovering that classification is just regression on probabilities)
"""


from typing import Optional

from foundations.activations import sigmoid
from foundations.linear_algebra.vectors_and_matrices import Matrix, Vector
from foundations.loss_functions.cross_entropy_loss import binary_cross_entropy


class LogisticRegression:
    """
    Binary logistic regression classifier.
    
    This implements binary classification using sigmoid activation and binary
    cross-entropy loss. The model learns a linear decision boundary that separates
    two classes in feature space.
    
    THE MODEL:
    ---------
    Given features x, the model predicts:
        P(y=1|x) = σ(w^T x) = 1 / (1 + e^(-w^T x))
    
    Where:
    - w are the learned weights
    - σ is the sigmoid function
    - P(y=1|x) is the probability that x belongs to class 1
    
    MAKING PREDICTIONS:
    ------------------
    - If P(y=1|x) ≥ threshold: predict class 1
    - If P(y=1|x) < threshold: predict class 0
    
    The default threshold is 0.5, but you can adjust it based on your problem.
    
    THE TRAINING OBJECTIVE:
    ----------------------
    Minimize: Loss = Binary Cross-Entropy + λ × L2 Penalty
    
    Binary Cross-Entropy:
        -(1/n) Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
    
    L2 Penalty (optional):
        λ Σⱼ wⱼ² (don't penalize bias w₀)
    
    THE GRADIENT:
    ------------
    ∂Loss/∂w = (1/n) X^T (ŷ - y) + 2λw
    
    The first term is the prediction error weighted by features (same as linear
    regression!). The second term is the regularization gradient.
    
    PARAMETERS:
    ----------
    learning_rate : float, default=0.1
        Step size for gradient descent
        Typically larger than linear regression (0.01-0.5) because sigmoid bounds outputs
        
    n_epochs : int, default=100
        Number of training iterations
        
    regularization : float, default=0.0
        L2 regularization strength (λ)
        - 0.0: No regularization
        - 0.01-0.1: Light regularization
        - 1.0-10.0: Strong regularization
        
    threshold : float, default=0.5
        Decision threshold for classification
        - 0.5: Default, balanced
        - > 0.5: Favor class 0 (higher precision for class 1)
        - < 0.5: Favor class 1 (higher recall for class 1)
        
    verbose : bool, default=False
        Whether to print training progress
        
    ATTRIBUTES:
    ----------
    weights_ : Vector
        Learned weights (including bias as first element)
        
    history_ : dict
        Training history with 'loss', 'accuracy', and 'epoch'
        
    EXAMPLE USAGE:
    -------------
    >>> from foundations.data_preprocessing.feature_engineering import (
    ...     Standardizer, add_bias_column, train_test_split
    ... )
    >>> 
    >>> # Prepare data
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> 
    >>> # Standardize (critical!)
    >>> scaler = Standardizer()
    >>> X_train_scaled = scaler.fit_transform(X_train)
    >>> X_test_scaled = scaler.transform(X_test)
    >>> 
    >>> # Add bias
    >>> X_train_final = add_bias_column(X_train_scaled)
    >>> X_test_final = add_bias_column(X_test_scaled)
    >>> 
    >>> # Train
    >>> model = LogisticRegression(learning_rate=0.1, n_epochs=500)
    >>> model.fit(X_train_final, y_train)
    >>> 
    >>> # Predict probabilities
    >>> probabilities = model.predict_proba(X_test_final)
    >>> 
    >>> # Predict classes
    >>> predictions = model.predict(X_test_final)
    >>> 
    >>> # Evaluate
    >>> accuracy = model.score(X_test_final, y_test)
    >>> print(f"Accuracy: {accuracy:.4f}")
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        n_epochs: int = 100,
        regularization: float = 0.0,
        threshold: float = 0.5,
        verbose: bool = False
    ):
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        
        if n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {n_epochs}")
        
        if regularization < 0:
            raise ValueError(f"regularization must be non-negative, got {regularization}")
        
        if not 0 < threshold < 1:
            raise ValueError(f"threshold must be in (0,1), got {threshold}")
        
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.regularization = regularization
        self.threshold = threshold
        self.verbose = verbose
        
        # Will be set during training
        self.weights_ = None
        self.history_ = None
    
    def fit(self, X: Matrix, y: Matrix) -> 'LogisticRegression':
        """
        Train logistic regression using gradient descent.
        
        THE TRAINING ALGORITHM:
        ----------------------
        Initialize weights to zeros
        
        For each epoch:
            # Forward pass
            z = Xw (linear combination)
            ŷ = σ(z) (sigmoid activation)
            
            # Compute loss
            Loss = -(1/n) Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)] + λ||w||²
            
            # Backward pass
            ∂Loss/∂w = (1/n) X^T (ŷ - y) + 2λw
            
            # Update
            w := w - α × ∂Loss/∂w
        
        NUMERICAL STABILITY:
        -------------------
        When computing log(ŷ), we need to be careful:
        - If ŷ = 0: log(0) = -∞ (numerical error!)
        - If ŷ = 1: log(1-1) = log(0) = -∞
        
        Solution: Clip predictions to [ε, 1-ε] where ε = 1e-15
        This prevents numerical overflow without meaningfully affecting the model.
        
        CONVERGENCE:
        -----------
        Cross-entropy loss is convex for logistic regression, so gradient descent
        is guaranteed to converge to the global minimum (given appropriate learning rate).
        
        Watch the loss during training - it should decrease smoothly. If it oscillates
        or increases, your learning rate is too large.
        
        PARAMETERS:
        ----------
        X : Matrix, shape (n_samples, n_features)
            Training features (should include bias column and be standardized!)
            
        y : Matrix, shape (n_samples, 1)
            Training labels (must be 0 or 1)
            
        RETURNS:
        -------
        self : LogisticRegression
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
            'loss': [],
            'accuracy': [],
            'epoch': []
        }
        
        if self.verbose:
            print("=" * 70)
            print("TRAINING LOGISTIC REGRESSION")
            print("=" * 70)
            print(f"Dataset: {n_samples} samples, {n_features} features")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Epochs: {self.n_epochs}")
            print(f"Regularization: {self.regularization}")
            print("-" * 70)
        
        # Training loop
        for epoch in range(self.n_epochs):
            # =================================================================
            # FORWARD PASS
            # =================================================================
            
            # Step 1: Compute linear combination z = Xw
            z_values = []
            for i in range(n_samples):
                x_i = X.row(i)
                z_i = x_i.dot(self.weights_)
                z_values.append(z_i)
            
            # Step 2: Apply sigmoid to get probabilities
            probabilities = []
            for z in z_values:
                prob = sigmoid(z)
                # Clip for numerical stability
                prob = max(1e-15, min(1 - 1e-15, prob))
                probabilities.append([prob])
            
            y_pred = Matrix(probabilities)
            
            # =================================================================
            # COMPUTE LOSS
            # =================================================================
            
            # Binary cross-entropy
            loss = binary_cross_entropy(y, y_pred)
            
            # Add L2 regularization (don't penalize bias!)
            if self.regularization > 0:
                l2_penalty = sum(self.weights_[j] ** 2 
                               for j in range(1, n_features))
                loss += self.regularization * l2_penalty
            
            # Compute accuracy for monitoring
            correct = sum(1 for i in range(n_samples)
                         if (y_pred[i, 0] >= self.threshold) == (y[i, 0] == 1))
            accuracy = correct / n_samples
            
            # =================================================================
            # BACKWARD PASS (COMPUTE GRADIENT)
            # =================================================================
            
            # Gradient: (1/n) X^T (ŷ - y)
            gradient_elements = []
            for j in range(n_features):
                feature_col = X.column(j)
                
                # Compute: Σᵢ xᵢⱼ (ŷᵢ - yᵢ)
                grad_j = 0.0
                for i in range(n_samples):
                    error = y_pred[i, 0] - y[i, 0]
                    grad_j += feature_col[i] * error
                
                # Average over samples
                grad_j /= n_samples
                
                # Add regularization gradient (except for bias)
                if j > 0 and self.regularization > 0:
                    grad_j += 2 * self.regularization * self.weights_[j]
                
                gradient_elements.append(grad_j)
            
            gradient = Vector(gradient_elements)
            
            # =================================================================
            # UPDATE WEIGHTS
            # =================================================================
            
            # w := w - α × ∇Loss
            updated_weights = []
            for j in range(n_features):
                new_weight = self.weights_[j] - self.learning_rate * gradient[j]
                updated_weights.append(new_weight)
            
            self.weights_ = Vector(updated_weights)
            
            # =================================================================
            # RECORD HISTORY
            # =================================================================
            
            self.history_['loss'].append(loss)
            self.history_['accuracy'].append(accuracy)
            self.history_['epoch'].append(epoch)
            
            # Print progress
            if self.verbose and (epoch % 50 == 0 or epoch == self.n_epochs - 1):
                print(f"Epoch {epoch:4d}: Loss = {loss:.6f}, Accuracy = {accuracy:.4f}")
        
        if self.verbose:
            print("-" * 70)
            print("TRAINING COMPLETE!")
            print(f"Final loss: {self.history_['loss'][-1]:.6f}")
            print(f"Final accuracy: {self.history_['accuracy'][-1]:.4f}")
            print("=" * 70)
        
        return self
    
    def predict_proba(self, X: Matrix) -> Matrix:
        """
        Predict class probabilities.
        
        This returns P(y=1|x) for each sample. The probability of class 0 is
        simply 1 - P(y=1|x).
        
        THE COMPUTATION:
        ---------------
        For each sample x:
            z = w^T x (linear combination)
            P(y=1|x) = σ(z) = 1 / (1 + e^(-z))
        
        INTERPRETING PROBABILITIES:
        --------------------------
        - P close to 1: Very confident it's class 1
        - P close to 0.5: Uncertain, near decision boundary
        - P close to 0: Very confident it's class 0
        
        The distance from the decision boundary (|w^T x|) indicates confidence.
        Larger |w^T x| means more confident prediction.
        
        PARAMETERS:
        ----------
        X : Matrix, shape (n_samples, n_features)
            Features to predict on
            
        RETURNS:
        -------
        probabilities : Matrix, shape (n_samples, 1)
            P(y=1|x) for each sample
        """
        
        if self.weights_ is None:
            raise RuntimeError("Model not trained! Call .fit() first.")
        
        if X.num_cols != len(self.weights_):
            raise ValueError(
                f"X has {X.num_cols} features but model expects {len(self.weights_)}"
            )
        
        probabilities = []
        for i in range(X.num_rows):
            x_i = X.row(i)
            z = x_i.dot(self.weights_)
            prob = sigmoid(z)
            probabilities.append([prob])
        
        return Matrix(probabilities)
    
    def predict(self, X: Matrix, threshold: Optional[float] = None) -> Matrix:
        """
        Predict class labels.
        
        This thresholds the probabilities to make hard predictions:
        - If P(y=1|x) ≥ threshold: predict 1
        - If P(y=1|x) < threshold: predict 0
        
        CHOOSING THE THRESHOLD:
        ----------------------
        The default threshold is 0.5 (predict class 1 if more likely than class 0).
        But you can adjust it based on your problem:
        
        Higher threshold (e.g., 0.7):
        - Predicts class 1 only when very confident
        - Increases precision (fewer false positives)
        - Decreases recall (more false negatives)
        - Use when false positives are costly
        
        Lower threshold (e.g., 0.3):
        - Predicts class 1 even when somewhat uncertain
        - Increases recall (fewer false negatives)
        - Decreases precision (more false positives)
        - Use when false negatives are costly
        
        Example: Cancer screening
        - False negative (miss cancer) is catastrophic
        - False positive (unnecessary testing) is acceptable
        - Use lower threshold to catch more cases
        
        Example: Spam filter
        - False positive (block real email) is very bad
        - False negative (let spam through) is acceptable
        - Use higher threshold to avoid blocking real emails
        
        PARAMETERS:
        ----------
        X : Matrix
            Features to predict on
            
        threshold : float, optional
            Decision threshold (uses self.threshold if not provided)
            
        RETURNS:
        -------
        predictions : Matrix, shape (n_samples, 1)
            Predicted class labels (0 or 1)
        """
        
        if threshold is None:
            threshold = self.threshold
        
        # Get probabilities
        probabilities = self.predict_proba(X)
        
        # Threshold to get class labels
        predictions = []
        for i in range(probabilities.num_rows):
            prob = probabilities[i, 0]
            label = 1.0 if prob >= threshold else 0.0
            predictions.append([label])
        
        return Matrix(predictions)
    
    def score(self, X: Matrix, y: Matrix, threshold: Optional[float] = None) -> float:
        """
        Compute classification accuracy.
        
        Accuracy = (number of correct predictions) / (total predictions)
        
        NOTE: Accuracy can be misleading for imbalanced datasets!
        If 95% of samples are class 0, predicting all class 0 gives 95% accuracy
        but is completely useless. Always look at precision, recall, and F1 score
        in addition to accuracy.
        
        PARAMETERS:
        ----------
        X : Matrix
            Features
            
        y : Matrix
            True labels (0 or 1)
            
        threshold : float, optional
            Decision threshold
            
        RETURNS:
        -------
        accuracy : float
            Fraction of correct predictions (between 0 and 1)
        """
        
        predictions = self.predict(X, threshold=threshold)
        
        correct = sum(1 for i in range(y.num_rows)
                     if abs(predictions[i, 0] - y[i, 0]) < 0.5)
        
        return correct / y.num_rows
    
    def get_decision_boundary_params(self) -> dict:
        """
        Get parameters defining the decision boundary.
        
        The decision boundary is where P(y=1|x) = 0.5, which occurs when w^T x = 0.
        
        For 2D data (two features x₁, x₂ plus bias):
            w₀ + w₁x₁ + w₂x₂ = 0
            x₂ = -(w₀ + w₁x₁) / w₂
        
        This is a line with:
            slope = -w₁ / w₂
            intercept = -w₀ / w₂
        
        The weight vector [w₁, w₂] is perpendicular to this line and points
        toward the class 1 region.
        
        RETURNS:
        -------
        params : dict
            Contains:
            - 'weights': The weight vector (without bias)
            - 'bias': The bias term (w₀)
            - 'slope': Slope of decision boundary (for 2D case)
            - 'intercept': Intercept of decision boundary (for 2D case)
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