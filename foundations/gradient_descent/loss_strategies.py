"""
LOSS STRATEGIES: PLUG-AND-PLAY OBJECTIVES
=========================================

This file implements the Strategy Pattern for loss functions.

WHY DO THIS?
-----------
The Gradient Descent algorithm doesn't actually care *what* it's minimizing.
It just needs to know two things:
1. "How bad are we doing?" (The Loss Scalar)
2. "Which way is downhill?" (The Gradient Vector)

By defining a strict interface (LossFunction), we can swap out MSE for MAE,
Cross-Entropy, or any custom loss without changing a single line of the 
optimizer code. This adheres to the Open-Closed Principle: open for extension
(new losses), but closed for modification (optimizers stay the same).
"""

from abc import ABC, abstractmethod

from foundations.linear_algebra.vectors_and_matrices import Matrix, Vector

# So see gradient descent would utlimately depend on the loss function we have so to make this generic , what I did was used Open-Closed Principle of programming , where I defined a rule ( interface ) for loss functions that tells what a loss function has ( in this case a loss and gradient functions ) , now the main optimizer is not concerned with loss or gradient caldulation it just knows that what ever is given to it follows this rule ( interface ) and based on that I can call to get the loss values wht we did was — the Strategy Pattern combined with polymorphism. 

class LossFunction(ABC):
    """
    The Contract. Any loss function used in our library MUST implement these methods.
    """
    
    @abstractmethod
    def calculate_loss(self, y_true: Matrix, y_pred: Matrix) -> float:
        """Calculate the scalar loss value (for monitoring progress)."""
        pass

    @abstractmethod
    def calculate_gradient(self, X: Matrix, y_true: Matrix, y_pred: Matrix, weights: Vector) -> Vector:
        """
        Calculate the gradient vector.
        
        This is the critical piece. It tells the optimizer how to change the weights
        to reduce the loss.
        """
        pass

## Few Example Loss functions

class MSELoss(LossFunction):
    """
    Mean Squared Error Strategy.
    Standard for Regression. Penalizes large errors heavily.
    """
    def calculate_loss(self, y_true: Matrix, y_pred: Matrix) -> float:
        n = y_true.num_rows
        # sum((y_hat - y)^2) / n
        errors = [(y_pred[i,0] - y_true[i,0])**2 for i in range(n)]
        return sum(errors) / n

    def calculate_gradient(self, X: Matrix, y_true: Matrix, y_pred: Matrix, weights: Vector) -> Vector:
        n_samples = X.num_rows
        n_features = X.num_cols
        
        # 1. Calculate Error Vector (y_pred - y_true)
        errors = []
        for i in range(n_samples):
            errors.append(y_pred[i,0] - y_true[i,0])
        error_vector = Vector(errors)
        
        # 2. Calculate Gradient: (2/n) * X^T @ error
        gradient_elements = []
        for j in range(n_features):
            feature_column = X.column(j) 
            # Dot product acts as the summation: sum(error_i * x_ij)
            grad_j = feature_column.dot(error_vector)
            gradient_elements.append((2 / n_samples) * grad_j)
            
        return Vector(gradient_elements)


class MAELoss(LossFunction):
    """
    Mean Absolute Error Strategy.
    Robust to outliers. Gradient is constant (sign of error).
    """
    def calculate_loss(self, y_true: Matrix, y_pred: Matrix) -> float:
        n = y_true.num_rows
        errors = [abs(y_pred[i,0] - y_true[i,0]) for i in range(n)]
        return sum(errors) / n

    def calculate_gradient(self, X: Matrix, y_true: Matrix, y_pred: Matrix, weights: Vector) -> Vector:
        n_samples = X.num_rows
        n_features = X.num_cols
        
        # 1. Calculate Sign Vector: sign(y_pred - y_true)
        signs = []
        for i in range(n_samples):
            diff = y_pred[i,0] - y_true[i,0]
            if diff > 0: signs.append(1.0)
            elif diff < 0: signs.append(-1.0)
            else: signs.append(0.0) # Subgradient at 0
        sign_vector = Vector(signs)
        
        # 2. Calculate Gradient: (1/n) * X^T @ sign_vector
        gradient_elements = []
        for j in range(n_features):
            feature_column = X.column(j)
            grad_j = feature_column.dot(sign_vector)
            gradient_elements.append(grad_j / n_samples)
            
        return Vector(gradient_elements)