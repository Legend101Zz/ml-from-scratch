"""
CROSS-ENTROPY LOSS: The Heart of Classification
===============================================

This module implements cross-entropy loss, which is THE standard loss function for
classification tasks. When I first encountered it, I found it confusing — why not
just use mean squared error like we did for regression?

Then I understood: Cross-entropy is derived from probability theory and information
theory. It's not just "a loss function that works" — it's the mathematically optimal
loss function for classification when you interpret predictions as probabilities!

MY LEARNING JOURNEY:
-------------------
I initially tried using MSE for binary classification:
- Model outputs: 0.2, 0.8, 0.6 (probabilities)
- True labels: 0, 1, 1
- MSE: Works but gradients are weird

Then I switched to cross-entropy:
- Same predictions and labels
- Cross-entropy: Gradients are clean and interpretable
- Training converged MUCH faster!

The key insight: Cross-entropy measures "surprise." If you predict high probability
for the correct class, loss is low. If you predict high probability for the WRONG
class, loss is high. It's like saying "How surprised would I be if this prediction
was correct?"

THE INFORMATION THEORY CONNECTION:
---------------------------------
Cross-entropy comes from Shannon's information theory. If an event has probability p,
the "information content" (surprise) is -log(p):

- Certain event (p=1): -log(1) = 0 bits of surprise (you expected it!)
- Coin flip (p=0.5): -log(0.5) = 1 bit of surprise (could go either way)
- Rare event (p=0.01): -log(0.01) = 6.6 bits of surprise (didn't see that coming!)

Cross-entropy averages this surprise over all your predictions. It asks: "On average,
how surprised am I by the true labels given my predicted probabilities?"

WHY IT'S BETTER THAN MSE FOR CLASSIFICATION:
--------------------------------------------
1. Probabilistic interpretation: Works naturally with probabilities
2. Better gradients: Doesn't saturate like MSE with sigmoid
3. Theoretical justification: Maximum likelihood estimation
4. Faster convergence: Gradients scale with confidence errors

WHAT WE'LL IMPLEMENT:
--------------------
1. Binary cross-entropy: For 2-class problems (yes/no, cat/dog)
2. Categorical cross-entropy: For multi-class problems (cat/dog/bird/...)
3. Sparse categorical cross-entropy: When labels are integers, not one-hot

"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from foundations.linear_algebra.vectors_and_matrices import Matrix, Vector
from typing import Union


# ==============================================================================
# BINARY CROSS-ENTROPY LOSS
# ==============================================================================

def binary_cross_entropy(
    y_true: Matrix,
    y_pred: Matrix,
    epsilon: float = 1e-15
) -> float:
    """
    Binary cross-entropy loss for binary classification.
    
    This is THE loss function for binary classification (two classes: 0 or 1,
    yes or no, cat or dog, etc.). It's used with logistic regression, neural
    networks for binary classification, and anywhere you're predicting probabilities
    for two classes.
    
    THE FORMULA:
    -----------
    BCE = -(1/n) Σ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]
    
    Where:
    - yᵢ ∈ {0, 1} is the true label (0 or 1)
    - ŷᵢ ∈ (0, 1) is the predicted probability
    - n is the number of samples
    
    UNDERSTANDING THE FORMULA:
    -------------------------
    Let's break down what happens for each case:
    
    Case 1: True label is 1 (positive class)
    - Formula becomes: -log(ŷ)
    - If ŷ = 0.9 (confident correct): -log(0.9) = 0.105 (low loss, good!)
    - If ŷ = 0.1 (confident wrong): -log(0.1) = 2.303 (high loss, bad!)
    
    Case 2: True label is 0 (negative class)
    - Formula becomes: -log(1-ŷ)
    - If ŷ = 0.1 (confident correct): -log(0.9) = 0.105 (low loss, good!)
    - If ŷ = 0.9 (confident wrong): -log(0.1) = 2.303 (high loss, bad!)
    
    So the loss penalizes confident wrong predictions heavily!
    
    WHY LOGARITHM?
    -------------
    The log comes from maximum likelihood! We want to maximize the probability
    of observing the true labels:
    
    L = Π P(yᵢ|xᵢ) = Π [ŷᵢʸⁱ · (1-ŷᵢ)¹⁻ʸⁱ]
    
    Taking log (easier to optimize sums than products):
    
    log(L) = Σ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]
    
    We want to MAXIMIZE log-likelihood, which is the same as MINIMIZING negative
    log-likelihood. And negative log-likelihood IS cross-entropy!
    
    So cross-entropy isn't arbitrary — it's the natural loss function that falls
    out of maximum likelihood estimation.
    
    NUMERICAL STABILITY:
    -------------------
    We can't compute log(0) or log(1) without issues (undefined or infinite).
    So we clip predictions to [epsilon, 1-epsilon]:
    
    ŷ_clipped = max(epsilon, min(1-epsilon, ŷ))
    
    This prevents numerical errors while barely affecting the loss.
    
    THE GRADIENT:
    ------------
    The beautiful thing about cross-entropy with sigmoid is the gradient:
    
    ∂BCE/∂z = (ŷ - y) / n
    
    Where z is the input to sigmoid (logit). The gradient is just the error!
    This is why logistic regression trains so well — clean, simple gradients.
    
    PARAMETERS:
    ----------
    y_true : Matrix, shape (n_samples, 1)
        True binary labels (0 or 1)
        
    y_pred : Matrix, shape (n_samples, 1)
        Predicted probabilities (between 0 and 1)
        Should be output of sigmoid activation!
        
    epsilon : float
        Small value to clip predictions for numerical stability
        
    RETURNS:
    -------
    loss : float
        Average binary cross-entropy loss
        
    EXAMPLE:
    -------
    >>> y_true = Matrix([[1], [0], [1], [1]])  # True labels
    >>> y_pred = Matrix([[0.9], [0.2], [0.8], [0.6]])  # Predictions from sigmoid
    >>> 
    >>> loss = binary_cross_entropy(y_true, y_pred)
    >>> print(f"Loss: {loss:.4f}")
    >>> # Low loss because predictions are mostly correct!
    """
    
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true is {y_true.shape}, y_pred is {y_pred.shape}"
        )
    
    if y_pred.num_cols != 1:
        raise ValueError(
            "binary_cross_entropy expects column vectors (n_samples, 1)"
        )
    
    n = y_true.num_rows
    total_loss = 0.0
    
    for i in range(n):
        y = y_true[i, 0]
        y_hat = y_pred[i, 0]
        
        # Clip predictions to prevent log(0) or log(1)
        y_hat_clipped = max(epsilon, min(1.0 - epsilon, y_hat))
        
        # Compute: -[y·log(ŷ) + (1-y)·log(1-ŷ)]
        if y == 1:
            # Positive class: -log(ŷ)
            loss_i = -math.log(y_hat_clipped)
        else:
            # Negative class: -log(1-ŷ)
            loss_i = -math.log(1.0 - y_hat_clipped)
        
        total_loss += loss_i
    
    # Average over all samples
    return total_loss / n


def binary_cross_entropy_gradient(
    y_true: Matrix,
    y_pred: Matrix
) -> Matrix:
    """
    Gradient of binary cross-entropy with respect to predictions.
    
    For logistic regression with sigmoid activation, this simplifies beautifully
    to just (ŷ - y)! This is one of the most elegant results in machine learning.
    
    THE DERIVATION:
    --------------
    Starting from BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
    
    Taking derivative with respect to ŷ:
    ∂BCE/∂ŷ = -[y/ŷ - (1-y)/(1-ŷ)]
            = -[y(1-ŷ) - (1-y)ŷ] / [ŷ(1-ŷ)]
            = -(y - ŷ) / [ŷ(1-ŷ)]
    
    Now, for logistic regression, ŷ = σ(z) where z = wᵀx.
    The chain rule gives us: ∂BCE/∂z = ∂BCE/∂ŷ · ∂ŷ/∂z
    
    We know: ∂σ/∂z = σ(1-σ) = ŷ(1-ŷ)
    
    So: ∂BCE/∂z = -(y - ŷ) / [ŷ(1-ŷ)] · ŷ(1-ŷ) = ŷ - y
    
    The ŷ(1-ŷ) terms cancel! This is why sigmoid and cross-entropy work so well
    together — they're mathematically designed to be paired.
    
    PARAMETERS:
    ----------
    y_true : Matrix
        True labels (0 or 1)
        
    y_pred : Matrix
        Predicted probabilities
        
    RETURNS:
    -------
    gradient : Matrix
        Gradient with respect to predictions (ŷ - y)
    """
    
    # Simple and elegant: gradient is just prediction error!
    gradient_data = []
    for i in range(y_true.num_rows):
        grad = y_pred[i, 0] - y_true[i, 0]
        gradient_data.append([grad])
    
    return Matrix(gradient_data)


# ==============================================================================
# CATEGORICAL CROSS-ENTROPY LOSS
# ==============================================================================

def categorical_cross_entropy(
    y_true: Matrix,
    y_pred: Matrix,
    epsilon: float = 1e-15
) -> float:
    """
    Categorical cross-entropy for multi-class classification.
    
    This generalizes binary cross-entropy to multiple classes! Use this when you
    have more than 2 classes (cat/dog/bird, digit 0-9, etc.).
    
    THE SETUP:
    ---------
    For K classes, we represent labels as one-hot vectors:
    - Class 0: [1, 0, 0, ..., 0]
    - Class 1: [0, 1, 0, ..., 0]
    - Class K-1: [0, 0, 0, ..., 1]
    
    Predictions are probability distributions (from softmax):
    - Example: [0.7, 0.2, 0.1] means 70% confident it's class 0
    
    THE FORMULA:
    -----------
    CCE = -(1/n) Σᵢ Σₖ yᵢₖ·log(ŷᵢₖ)
    
    Where:
    - yᵢₖ is 1 if sample i is class k, 0 otherwise (one-hot)
    - ŷᵢₖ is predicted probability for sample i being class k
    
    SIMPLIFIED:
    ----------
    Since yᵢₖ is one-hot (only one 1, rest zeros), the sum simplifies to:
    
    CCE = -(1/n) Σᵢ log(ŷᵢ,cᵢ)
    
    Where cᵢ is the true class of sample i. So we only look at the predicted
    probability for the TRUE class!
    
    INTERPRETATION:
    --------------
    Example: True class is 0 (one-hot: [1, 0, 0])
    - Predictions: [0.8, 0.1, 0.1] → Loss = -log(0.8) = 0.223 (good!)
    - Predictions: [0.2, 0.5, 0.3] → Loss = -log(0.2) = 1.609 (bad!)
    
    The loss only cares about the probability assigned to the correct class.
    High probability for correct class → low loss. Low probability → high loss.
    
    RELATIONSHIP TO SOFTMAX:
    -----------------------
    Categorical cross-entropy is typically used with softmax activation:
    
    z = Wx (linear layer)
    ŷ = softmax(z) (activation)
    Loss = categorical_cross_entropy(y, ŷ)
    
    The gradient through softmax + cross-entropy is beautifully simple:
    ∂Loss/∂z = (ŷ - y) / n
    
    Again, just the prediction error! This is why softmax and categorical
    cross-entropy are always used together.
    
    PARAMETERS:
    ----------
    y_true : Matrix, shape (n_samples, n_classes)
        True labels as one-hot vectors
        Each row has exactly one 1, rest are 0s
        
    y_pred : Matrix, shape (n_samples, n_classes)
        Predicted probabilities from softmax
        Each row sums to 1
        
    epsilon : float
        Clipping value for numerical stability
        
    RETURNS:
    -------
    loss : float
        Average categorical cross-entropy loss
        
    EXAMPLE:
    -------
    >>> # 3 samples, 4 classes
    >>> y_true = Matrix([
    ...     [1, 0, 0, 0],  # Sample 0 is class 0
    ...     [0, 1, 0, 0],  # Sample 1 is class 1
    ...     [0, 0, 0, 1]   # Sample 2 is class 3
    ... ])
    >>> 
    >>> y_pred = Matrix([
    ...     [0.7, 0.2, 0.05, 0.05],  # Confident in class 0 (correct!)
    ...     [0.1, 0.6, 0.2, 0.1],    # Confident in class 1 (correct!)
    ...     [0.3, 0.3, 0.3, 0.1]     # Not confident (wrong!)
    ... ])
    >>> 
    >>> loss = categorical_cross_entropy(y_true, y_pred)
    >>> # First two predictions have low loss, third has high loss
    """
    
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true is {y_true.shape}, y_pred is {y_pred.shape}"
        )
    
    n_samples = y_true.num_rows
    n_classes = y_true.num_cols
    
    total_loss = 0.0
    
    for i in range(n_samples):
        # For each sample, sum over all classes
        sample_loss = 0.0
        
        for j in range(n_classes):
            y = y_true[i, j]
            y_hat = y_pred[i, j]
            
            # Clip for numerical stability
            y_hat_clipped = max(epsilon, min(1.0 - epsilon, y_hat))
            
            # Only add to loss if true label is 1 (one-hot)
            if y > 0:  # This will be 1 for the true class, 0 otherwise
                sample_loss -= y * math.log(y_hat_clipped)
        
        total_loss += sample_loss
    
    return total_loss / n_samples


def sparse_categorical_cross_entropy(
    y_true: Matrix,
    y_pred: Matrix,
    epsilon: float = 1e-15
) -> float:
    """
    Categorical cross-entropy when labels are integers, not one-hot.
    
    This is a convenience function for when you have integer labels instead of
    one-hot vectors. It's more memory efficient!
    
    WHEN TO USE:
    -----------
    Use categorical_cross_entropy when: y_true = [[1,0,0], [0,1,0]]
    Use sparse_categorical_cross_entropy when: y_true = [[0], [1]]
    
    Both compute the same loss, but sparse version is more convenient when you
    have class indices rather than one-hot vectors.
    
    THE FORMULA:
    -----------
    Same as categorical_cross_entropy, but we don't need to one-hot encode!
    
    Loss = -(1/n) Σᵢ log(ŷᵢ,yᵢ)
    
    Where yᵢ is the integer class label (0, 1, 2, ..., K-1).
    
    PARAMETERS:
    ----------
    y_true : Matrix, shape (n_samples, 1)
        Integer class labels (0 to n_classes-1)
        
    y_pred : Matrix, shape (n_samples, n_classes)
        Predicted probabilities from softmax
        
    epsilon : float
        Clipping value
        
    RETURNS:
    -------
    loss : float
        Average cross-entropy loss
        
    EXAMPLE:
    -------
    >>> y_true = Matrix([[0], [1], [2]])  # Class indices
    >>> y_pred = Matrix([
    ...     [0.7, 0.2, 0.1],  # High prob for class 0 (correct!)
    ...     [0.1, 0.8, 0.1],  # High prob for class 1 (correct!)
    ...     [0.3, 0.4, 0.3]   # Unsure about class 2 (correct class has low prob!)
    ... ])
    >>> 
    >>> loss = sparse_categorical_cross_entropy(y_true, y_pred)
    >>> # First two have low loss, third has higher loss
    """
    
    if y_true.num_cols != 1:
        raise ValueError(
            "sparse_categorical_cross_entropy expects y_true to be column vector"
        )
    
    n_samples = y_true.num_rows
    n_classes = y_pred.num_cols
    
    total_loss = 0.0
    
    for i in range(n_samples):
        # Get true class index
        true_class = int(y_true[i, 0])
        
        if true_class < 0 or true_class >= n_classes:
            raise ValueError(
                f"Class index {true_class} is out of range [0, {n_classes})"
            )
        
        # Get predicted probability for true class
        y_hat = y_pred[i, true_class]
        
        # Clip for numerical stability
        y_hat_clipped = max(epsilon, min(1.0 - epsilon, y_hat))
        
        # Compute loss: -log(ŷ_true_class)
        loss_i = -math.log(y_hat_clipped)
        total_loss += loss_i
    
    return total_loss / n_samples


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def one_hot_encode(class_indices: Matrix, n_classes: int) -> Matrix:
    """
    Convert integer class labels to one-hot encoded vectors.
    
    This is useful when you have integer labels but need one-hot vectors for
    categorical cross-entropy.
    
    THE TRANSFORMATION:
    ------------------
    Input: [0, 2, 1] (class indices)
    Output: [[1, 0, 0],
             [0, 0, 1],
             [0, 1, 0]] (one-hot vectors)
    
    PARAMETERS:
    ----------
    class_indices : Matrix, shape (n_samples, 1)
        Integer class labels
        
    n_classes : int
        Total number of classes
        
    RETURNS:
    -------
    one_hot : Matrix, shape (n_samples, n_classes)
        One-hot encoded vectors
        
    EXAMPLE:
    -------
    >>> labels = Matrix([[0], [2], [1]])
    >>> one_hot = one_hot_encode(labels, n_classes=3)
    >>> print(one_hot.elements)
    [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    """
    
    if class_indices.num_cols != 1:
        raise ValueError("class_indices must be column vector")
    
    n_samples = class_indices.num_rows
    one_hot_data = []
    
    for i in range(n_samples):
        class_idx = int(class_indices[i, 0])
        
        if class_idx < 0 or class_idx >= n_classes:
            raise ValueError(
                f"Class index {class_idx} is out of range [0, {n_classes})"
            )
        
        # Create one-hot vector: all zeros except 1 at class_idx
        row = [0.0] * n_classes
        row[class_idx] = 1.0
        one_hot_data.append(row)
    
    return Matrix(one_hot_data)


def one_hot_decode(one_hot: Matrix) -> Matrix:
    """
    Convert one-hot encoded vectors back to integer class labels.
    
    This is the inverse of one_hot_encode.
    
    THE TRANSFORMATION:
    ------------------
    Input: [[1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]] (one-hot vectors)
    Output: [0, 2, 1] (class indices)
    
    PARAMETERS:
    ----------
    one_hot : Matrix, shape (n_samples, n_classes)
        One-hot encoded vectors
        
    RETURNS:
    -------
    class_indices : Matrix, shape (n_samples, 1)
        Integer class labels
    """
    
    n_samples = one_hot.num_rows
    n_classes = one_hot.num_cols
    
    indices_data = []
    
    for i in range(n_samples):
        # Find the index of the 1 in this row
        class_idx = None
        for j in range(n_classes):
            if one_hot[i, j] == 1.0:
                class_idx = j
                break
        
        if class_idx is None:
            raise ValueError(f"Row {i} is not a valid one-hot vector")
        
        indices_data.append([float(class_idx)])
    
    return Matrix(indices_data)


# ==============================================================================
# CROSS-ENTROPY LOSS STRATEGY (for gradient descent optimizer)
# ==============================================================================

class BinaryCrossEntropyLoss:
    """
    Binary cross-entropy loss as a strategy object for gradient descent.
    
    This wraps binary cross-entropy in the LossFunction interface so it can be
    used with our gradient descent optimizers.
    """
    
    def calculate_loss(self, y_true: Matrix, y_pred: Matrix) -> float:
        """Compute binary cross-entropy loss."""
        return binary_cross_entropy(y_true, y_pred)
    
    def calculate_gradient(
        self,
        X: Matrix,
        y_true: Matrix,
        y_pred: Matrix,
        weights: Vector
    ) -> Vector:
        """
        Compute gradient for logistic regression.
        
        For logistic regression with sigmoid + binary cross-entropy,
        the gradient simplifies to: (1/n) Xᵀ(ŷ - y)
        """
        
        n_samples = X.num_rows
        n_features = X.num_cols
        
        # Compute errors: ŷ - y
        errors = []
        for i in range(n_samples):
            error = y_pred[i, 0] - y_true[i, 0]
            errors.append(error)
        
        error_vector = Vector(errors)
        
        # Compute gradient: (1/n) Xᵀ @ errors
        gradient_elements = []
        for j in range(n_features):
            feature_col = X.column(j)
            grad_j = feature_col.dot(error_vector)
            gradient_elements.append(grad_j / n_samples)
        
        return Vector(gradient_elements)


# ==============================================================================
# EXAMPLE USAGE AND TESTING
# ==============================================================================

if __name__ == "__main__":
    """
    Demonstrate cross-entropy loss functions with examples.
    """
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║      CROSS-ENTROPY LOSS: THE MATHEMATICS OF CLASSIFICATION      ║
║                                                                  ║
║  "Why can't I use MSE for classification? Oh... THAT'S why!"   ║
║  - Me, after implementing cross-entropy                         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # =========================================================================
    # Demo 1: Binary Cross-Entropy
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("DEMO 1: BINARY CROSS-ENTROPY")
    print("=" * 70)
    
    print("\nScenario: Binary classification (spam vs not-spam)")
    
    y_true_binary = Matrix([[1], [0], [1], [1], [0]])
    
    # Good predictions
    y_pred_good = Matrix([[0.9], [0.1], [0.85], [0.95], [0.05]])
    loss_good = binary_cross_entropy(y_true_binary, y_pred_good)
    
    # Bad predictions
    y_pred_bad = Matrix([[0.3], [0.7], [0.4], [0.2], [0.8]])
    loss_bad = binary_cross_entropy(y_true_binary, y_pred_bad)
    
    print("\nTrue labels: [1, 0, 1, 1, 0]")
    print("\nGood predictions: [0.9, 0.1, 0.85, 0.95, 0.05]")
    print(f"Loss: {loss_good:.4f} (low - predictions match truth!)")
    
    print("\nBad predictions: [0.3, 0.7, 0.4, 0.2, 0.8]")
    print(f"Loss: {loss_bad:.4f} (high - predictions are wrong!)")
    
    print(f"\nLoss ratio: {loss_bad/loss_good:.2f}x")
    print("Bad predictions have much higher loss!")
    
    # Show individual contributions
    print("\nBreakdown of good predictions:")
    print("  True | Pred | Loss      | Interpretation")
    print("-------|------|-----------|----------------------------------")
    for i in range(y_true_binary.num_rows):
        y = y_true_binary[i, 0]
        y_hat = y_pred_good[i, 0]
        
        # Compute individual loss
        if y == 1:
            loss_i = -math.log(max(1e-15, y_hat))
        else:
            loss_i = -math.log(max(1e-15, 1 - y_hat))
        
        interp = "Confident & correct" if loss_i < 0.2 else "Uncertain or wrong"
        print(f"   {y:.0f}   | {y_hat:.2f} | {loss_i:.4f}    | {interp}")
    
    # =========================================================================
    # Demo 2: Categorical Cross-Entropy
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("DEMO 2: CATEGORICAL CROSS-ENTROPY (Multi-class)")
    print("=" * 70)
    
    print("\nScenario: 3-class classification (cat/dog/bird)")
    
    # One-hot encoded labels
    y_true_cat = Matrix([
        [1, 0, 0],  # cat
        [0, 1, 0],  # dog
        [0, 0, 1],  # bird
        [1, 0, 0]   # cat
    ])
    
    # Good predictions (confident in correct class)
    y_pred_cat_good = Matrix([
        [0.8, 0.15, 0.05],  # Confident cat (correct!)
        [0.1, 0.7, 0.2],    # Confident dog (correct!)
        [0.2, 0.1, 0.7],    # Confident bird (correct!)
        [0.85, 0.1, 0.05]   # Confident cat (correct!)
    ])
    
    # Bad predictions (confused)
    y_pred_cat_bad = Matrix([
        [0.3, 0.4, 0.3],   # Thinks it's dog (wrong!)
        [0.4, 0.3, 0.3],   # Unsure (sort of wrong)
        [0.4, 0.4, 0.2],   # Thinks it's cat/dog (wrong!)
        [0.2, 0.5, 0.3]    # Thinks it's dog (wrong!)
    ])
    
    loss_cat_good = categorical_cross_entropy(y_true_cat, y_pred_cat_good)
    loss_cat_bad = categorical_cross_entropy(y_true_cat, y_pred_cat_bad)
    
    print("\nGood predictions (confident in correct class):")
    print(f"Loss: {loss_cat_good:.4f}")
    
    print("\nBad predictions (confused about classes):")
    print(f"Loss: {loss_cat_bad:.4f}")
    
    print(f"\nLoss ratio: {loss_cat_bad/loss_cat_good:.2f}x")
    
    # =========================================================================
    # Demo 3: Sparse Categorical Cross-Entropy
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("DEMO 3: SPARSE CATEGORICAL CROSS-ENTROPY")
    print("=" * 70)
    
    print("\nSame as categorical, but labels are integers instead of one-hot")
    
    # Integer labels
    y_true_sparse = Matrix([[0], [1], [2], [0]])  # cat, dog, bird, cat
    
    loss_sparse_good = sparse_categorical_cross_entropy(y_true_sparse, y_pred_cat_good)
    loss_sparse_bad = sparse_categorical_cross_entropy(y_true_sparse, y_pred_cat_bad)
    
    print(f"\nGood predictions: Loss = {loss_sparse_good:.4f}")
    print(f"Bad predictions: Loss = {loss_cat_bad:.4f}")
    
    print("\n✓ Sparse version gives same loss as categorical!")
    print("  (Just more convenient when you have integer labels)")
    
    # =========================================================================
    # Demo 4: One-Hot Encoding
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("DEMO 4: ONE-HOT ENCODING")
    print("=" * 70)
    
    labels_int = Matrix([[0], [2], [1], [0]])
    labels_onehot = one_hot_encode(labels_int, n_classes=3)
    labels_decoded = one_hot_decode(labels_onehot)
    
    print("\nInteger labels:")
    print(f"  {[int(labels_int[i,0]) for i in range(labels_int.num_rows)]}")
    
    print("\nOne-hot encoded:")
    for i in range(labels_onehot.num_rows):
        print(f"  {labels_onehot.elements[i]}")
    
    print("\nDecoded back to integers:")
    print(f"  {[int(labels_decoded[i,0]) for i in range(labels_decoded.num_rows)]}")
    
    print("\n✓ Encoding and decoding are inverse operations!")
    
    # =========================================================================
    # Demo 5: Why Cross-Entropy vs MSE for Classification
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("DEMO 5: WHY CROSS-ENTROPY INSTEAD OF MSE?")
    print("=" * 70)
    
    y_true_demo = Matrix([[1], [0]])
    y_pred_demo = Matrix([[0.9], [0.1]])  # Good predictions
    
    # Cross-entropy loss
    ce_loss = binary_cross_entropy(y_true_demo, y_pred_demo)
    
    # MSE loss (for comparison)
    mse_loss = sum((y_pred_demo[i,0] - y_true_demo[i,0])**2 for i in range(2)) / 2
    
    print("\nTrue labels: [1, 0]")
    print("Predictions: [0.9, 0.1]")
    print()
    print(f"Cross-entropy loss: {ce_loss:.4f}")
    print(f"MSE loss: {mse_loss:.4f}")
    
    print("\nNow with confident WRONG predictions:")
    y_pred_wrong = Matrix([[0.1], [0.9]])  # Completely wrong!
    
    ce_loss_wrong = binary_cross_entropy(y_true_demo, y_pred_wrong)
    mse_loss_wrong = sum((y_pred_wrong[i,0] - y_true_demo[i,0])**2 for i in range(2)) / 2
    
    print("\nPredictions: [0.1, 0.9] (wrong!)")
    print(f"Cross-entropy loss: {ce_loss_wrong:.4f}")
    print(f"MSE loss: {mse_loss_wrong:.4f}")
    
    print("\nCross-entropy penalty ratio:", f"{ce_loss_wrong/ce_loss:.2f}x")
    print("MSE penalty ratio:", f"{mse_loss_wrong/mse_loss:.2f}x")
    
    print("\n→ Cross-entropy penalizes confident wrong predictions MORE!")
    print("  This is why it's better for classification.")
    
    # =========================================================================
    # Key Takeaways
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. BINARY CROSS-ENTROPY: For 2-class problems
   - Formula: -[y·log(ŷ) + (1-y)·log(1-ŷ)]
   - Use with sigmoid activation
   - Gradient with sigmoid: just (ŷ - y)!

2. CATEGORICAL CROSS-ENTROPY: For multi-class problems
   - Formula: -Σ yₖ·log(ŷₖ)
   - Use with softmax activation
   - Labels must be one-hot encoded
   - Gradient with softmax: just (ŷ - y)!

3. SPARSE CATEGORICAL: Convenience for integer labels
   - Same as categorical, but labels are integers
   - More memory efficient
   - Easier to use when you have class indices

4. WHY NOT MSE FOR CLASSIFICATION?
   - MSE assumes outputs are continuous values
   - Cross-entropy assumes outputs are probabilities
   - Cross-entropy gradients are cleaner
   - Cross-entropy converges faster
   - Cross-entropy has theoretical justification (maximum likelihood)

5. THE BEAUTIFUL SYMMETRY:
   - Sigmoid + Binary CE → Gradient = ŷ - y
   - Softmax + Categorical CE → Gradient = ŷ - y
   This is NOT a coincidence — they're mathematically designed to pair!

6. PRACTICAL TIPS:
   - Always clip predictions to prevent log(0)
   - Make sure predictions sum to 1 (for softmax)
   - Use cross-entropy for classification, MSE for regression
   - Monitor loss during training (should decrease!)
   
The key insight: Cross-entropy isn't arbitrary — it's the natural loss
function that emerges from maximum likelihood estimation when you treat
model outputs as probabilities!
    """)