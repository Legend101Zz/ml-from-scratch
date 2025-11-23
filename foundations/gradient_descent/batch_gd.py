"""
THE BIG IDEA:
------------
Imagine you're lost in fog on a mountain, trying to reach the lowest valley.
You can't see far, but you can feel the slope beneath your feet.

Strategy:
1. Feel which direction slopes downward (compute the gradient)
2. Take a small step in that direction (update parameters)
3. Repeat until you reach the bottom (converge to minimum loss)

This is EXACTLY what Batch Gradient Descent does:
- The "mountain" is the loss function (how wrong our predictions are)
- The "coordinates" are our model parameters (weights)
- The "slope" is the gradient (computed via calculus)
- Each "step" is a parameter update

THE MATHEMATICS:
---------------
We want to minimize the loss function , which is let's say in this case the Mean Squared Error loss:

    J(w) = (1/n) Σᵢ (ŷᵢ - yᵢ)²
    
Where:
- w = weight vector (parameters we're optimizing)
- ŷᵢ = prediction for sample i = xᵢᵀw (dot product)
- yᵢ = true label for sample i
- n = number of samples

The gradient tells us how much the loss increases when we change w:

    ∇J(w) = (2/n) Σᵢ (ŷᵢ - yᵢ) · xᵢ
          = (2/n) Xᵀ(ŷ - y)
          
In matrix form:
- X is the data matrix (n × d): each row is a sample, each column is a feature
- ŷ is the prediction vector (n × 1): predictions for all samples
- y is the true label vector (n × 1): true values for all samples

The update rule moves OPPOSITE to the gradient (to minimize loss):

    w_new = w_old - α · ∇J(w)
    
Where α (alpha) is the learning rate — how big each step is.

WHY "BATCH"?
-----------
"Batch" means we use ALL samples to compute the gradient before updating.
We average the gradient contributions from every single data point.

This is the most accurate gradient estimate but also the slowest:
- Pro: Smooth, deterministic convergence
- Pro: Guaranteed to reach global minimum (for convex functions)
- Con: Slow for large datasets (must process all data per update)
- Con: Can get stuck in local minima (for non-convex functions)

"""



from foundations.linear_algebra.vectors_and_matrices import Matrix, Vector

from .loss_strategies import LossFunction


class MSELoss(LossFunction):
    """
    Mean Squared Error Strategy.
    Loss = (1/n) * Σ(y_pred - y_true)^2
    Gradient = (2/n) * X^T · (y_pred - y_true)
    """    
    def calculate_loss(self, y_true, y_pred):
        n = y_true.num_rows
        errors = [(y_pred[i,0] - y_true[i,0])**2 for i in range(n)]
        return sum(errors) / n
    
    def calculate_gradient(self, X, y_true, y_pred, weights):
        n_samples = X.num_rows
        n_features = X.num_cols
        
        #1. Calculate Error Vector ( y_pred - y_true )
        errors = []
        for i in range(n_samples):
            errors.append(y_pred[i,0] - y_true[i,0])
        errors_vector = Vector(errors)
        
        #2. Calculate Gradient: (2/n) * X^T @ error
        # We don't have a transpose method, so we iterate features (columns of X)
        gradient_elements = []
        for j in range(n_features):
            # Extract feature column j as a vector
            feature_column = X.column(j) 
            
            # Dot product of Feature Column and Error Vector
            # This corresponds to: sum(error_i * x_ij)
            grad_j = feature_column.dot(errors_vector)
            
            # Average and scale (the 2 comes from derivative of square)
            gradient_elements.append((2 / n_samples) * grad_j)
            
        return Vector(gradient_elements)
    
class MAELoss(LossFunction):
    """
    Mean Absolute Error Strategy.
    Loss = (1/n) * Σ|y_pred - y_true|
    Gradient = (1/n) * X^T · sign(y_pred - y_true)
    
    Note: This is robust to outliers but has a discontinuous gradient at 0.
    """
    def calculate_loss(self, y_true: Matrix, y_pred: Matrix) -> float:
        n = y_true.num_rows
        errors = [abs(y_pred[i,0] - y_true[i,0]) for i in range(n)]
        return sum(errors) / n

    def calculate_gradient(self, X: Matrix, y_true: Matrix, y_pred: Matrix, weights: Vector) -> Vector:
        n_samples = X.num_rows
        n_features = X.num_cols
        
        # 1. Calculate Sign Vector: sign(y_pred - y_true)
        # +1 if prediction is too high, -1 if too low
        signs = []
        for i in range(n_samples):
            diff = y_pred[i,0] - y_true[i,0]
            if diff > 0: signs.append(1.0)
            elif diff < 0: signs.append(-1.0)
            else: signs.append(0.0) # Subgradient at 0
        sign_vector = Vector(signs)
        
        # 2. Calculate Gradient: (1/n) * X.T @ sign_vector
        gradient_elements = []
        for j in range(n_features):
            feature_column = X.column(j)
            grad_j = feature_column.dot(sign_vector)
            gradient_elements.append(grad_j / n_samples)
            
        return Vector(gradient_elements)

class BatchGradientDescent:
    """
    A generic optimizer that minimizes ANY differentiable loss function.
    
    INTUITION:
    ---------
    Think of training as rolling a ball down a hill:
    - The "hill" is the loss surface (high loss = high elevation)
    - The "ball" is our current weight vector
    - Gravity pulls the ball downward = gradient pulls weights toward lower loss
    - Learning rate controls how fast the ball rolls
    
    At each step, we:
    1. Compute predictions for ALL samples
    2. Compute errors for ALL samples
    3. Average the gradients from ALL samples
    4. Take ONE step in that averaged direction
    
    ALGORITHM:
    ---------
    Initialize weights w to zeros
    For each epoch (1 to n_epochs):
        1. Compute predictions: ŷ = X @ w (matrix-vector product)
        2. Compute errors: errors = ŷ - y (element-wise subtraction)
        3. Compute gradient
        4. Update weights: w = w - α * ∇J (move downhill)
    Return optimized weights w
    
    PARAMETERS:
    ----------
    learning_rate : float
        Step size for parameter updates (α in the math).
        - Too small: Slow convergence (takes forever)
        - Too large: Divergence (overshoots minimum, goes crazy)
        - Typical values: 0.001 to 0.1
        
    n_epochs : int
        Number of complete passes through the dataset.
        Each epoch processes all n samples once.
        More epochs = more learning opportunities
        
    verbose : bool
        If True, print loss and weights during training.
        Useful for debugging and understanding convergence.
        
    ATTRIBUTES (learned during training):
    ----------
    weights_ : Vector
        The optimized weight vector after training.
        Shape: (n_features,)
        
    history_ : dict
        Training history containing:
        - 'loss': list of loss values per epoch (monitors convergence)
        - 'epoch': list of epoch numbers (for plotting)
    """
    def __init__(self,loss_function:LossFunction, learning_rate: float = 0.01, n_epochs: int = 100, verbose: bool =False):
        self.loss_function = loss_function # The strategy 
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights_ = None
        self.history_ = { 'loss':[], 'epoch':[] }
        self.verbose = verbose
        
    def fit(self, X:Matrix, y: Matrix) -> 'BatchGradientDescent':
        n_samples = X.num_rows
        n_features = X.num_cols
        
        #1. Initialize Weights ( Zeros )
        # Works for n features naturally because we create a vector of size n_features
        self.weights_ = Vector([0.0] * n_features)
        
        # 2. The Descent loop
        for epoch in range(self.n_epochs):

            # A. Forward Pass: Compute Predictions
            # y_pred = X @ weights
            predictions = []
            for i in range(n_samples):
                # Dot product of sample row and weights vector
                # w0*x0 + w1*x1 ... + wn*xn
                pred = X.row(i).dot(self.weights_)
                predictions.append([pred])
            y_pred = Matrix(predictions)
        
            # B. Compute Gradient (Delegated to Loss Strategy)
            # This is the key! The optimizer asks the loss function:
            # "Based on these predictions, which way is 'down'?"
            gradient = self.loss_function.calculate_gradient(X, y, y_pred, self.weights_)
            
            # C. Update Weights
            # w_new = w_old - learning_rate * gradient
            # We do this component-wise
            new_w_elements = []
            for j in range(n_features):
                update = self.weights_[j] - (self.learning_rate * gradient[j])
                new_w_elements.append(update)
            self.weights_ = Vector(new_w_elements)
            
            # D. Track Loss
            if self.verbose or epoch == self.n_epochs - 1:
                current_loss = self.loss_function.calculate_loss(y, y_pred)
                self.history_['loss'].append(current_loss)
                self.history_['epoch'].append(epoch)
                
                if self.verbose and epoch % (self.n_epochs // 10) == 0:
                    print(f"Epoch {epoch}: Loss {current_loss:.4f}")

        return self
    
    def predict(self, X: Matrix) -> Matrix:
        if self.weights_ is None:
            raise Exception("Not trained yet")
        
        predictions = []
        for i in range(X.num_rows):
            pred = X.row(i).dot(self.weights_)
            predictions.append([pred])
        return Matrix(predictions)
    
    
# ==============================================================================
# 4. DEMONSTRATION
# ==============================================================================

if __name__ == "__main__":
    # 1. Create Dummy Data (y = 2x1 + 3x2 + 1)
    X = Matrix([
        [1, 1, 1], 
        [1, 2, 1], 
        [1, 1, 2]
    ]) # Bias, Feat1, Feat2
    
    y = Matrix([
        [6],  # 1 + 2(1) + 3(1) = 6
        [8],  # 1 + 2(2) + 3(1) = 8
        [9]   # 1 + 2(1) + 3(2) = 9
    ])

    print("--- Testing with MSE Loss ---")
    mse_opt = BatchGradientDescent(loss_function=MSELoss(), learning_rate=0.1, n_epochs=200, verbose=True)
    w_mse = mse_opt.fit(X, y)
    print(f"MSE Learned Weights: {w_mse} (Target: [1, 2, 3])")

    print("\n--- Testing with MAE Loss ---")
    # MAE usually needs smaller learning rate near convergence due to constant gradient
    mae_opt = BatchGradientDescent(loss_function=MAELoss(), learning_rate=0.05, n_epochs=200, verbose=True)
    w_mae = mae_opt.fit(X, y)
    print(f"MAE Learned Weights: {w_mae} (Target: [1, 2, 3])")