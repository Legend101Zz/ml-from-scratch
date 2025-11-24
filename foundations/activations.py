"""
ACTIVATION FUNCTIONS: The Non-Linearity That Makes ML Powerful
==============================================================

This module implements activation functions - the secret sauce that makes neural
networks and many ML models actually work!

MY JOURNEY WITH ACTIVATIONS:
----------------------------
When I first implemented linear regression, I thought "Great! I can predict numbers."
Then I tried to predict categories (yes/no, cat/dog/bird) and hit a wall.

The problem: Linear models output ANY number (-∞ to +∞), but I needed:
- Probabilities for classification (0 to 1)
- Multi-class probabilities that sum to 1
- Non-linear transformations for complex patterns

That's when I discovered activation functions! They're simple mathematical functions
that transform model outputs into useful ranges or add non-linearity.

THE BIG INSIGHT:
---------------
Linear models can only learn linear relationships (straight lines, flat planes).
But the real world is full of curves and complex patterns!

Activation functions add non-linearity between layers. Stack enough linear layers
with activations between them, and you can approximate ANY function! This is the
universal approximation theorem, and it's why deep learning works.

WHAT WE'LL IMPLEMENT:
--------------------
1. Sigmoid: σ(x) = 1/(1+e^(-x))
   - Squashes any number to (0, 1)
   - Used in binary classification
   - Interprets output as probability

2. Softmax: Generalization of sigmoid to multiple classes
   - Outputs sum to 1
   - Used in multi-class classification
   
3. ReLU: max(0, x)
   - Most popular activation in deep learning
   - Simple but powerful
   - Prevents vanishing gradients

4. Tanh: (e^x - e^(-x))/(e^x + e^(-x))
   - Squashes to (-1, 1)
   - Zero-centered (better than sigmoid)
   
5. Step function: For perceptron (historical importance)

WHY THESE MATTER:
----------------
Without activations, stacking multiple linear layers would just create... another
linear layer! The activations are what give neural networks their power to learn
complex, non-linear patterns.

"""

import math
from typing import  Union
from foundations.linear_algebra.vectors_and_matrices import Vector, Matrix


# ==============================================================================
# SIGMOID ACTIVATION
# ==============================================================================

def sigmoid(z: Union[float, Vector, Matrix]) -> Union[float, Vector, Matrix]:
    """
    Sigmoid activation function: σ(z) = 1/(1+e^(-z))
    
    This is THE activation function for binary classification! It takes any real
    number and squashes it into the range (0, 1), which we can interpret as a
    probability.
    
    THE MATHEMATICAL BEAUTY:
    -----------------------
    The sigmoid has several beautiful properties:
    
    1. BOUNDED OUTPUT: Always between 0 and 1
       - z = -∞ → σ(z) = 0
       - z = 0 → σ(z) = 0.5
       - z = +∞ → σ(z) = 1
    
    2. SMOOTH AND DIFFERENTIABLE:
       The derivative is: σ'(z) = σ(z)(1 - σ(z))
       This makes gradient descent work beautifully!
    
    3. S-SHAPED CURVE (sigmoid means "S-shaped"):
       Gentle transition from 0 to 1, not abrupt
    
    THE PROBABILITY INTERPRETATION:
    ------------------------------
    In logistic regression, we compute z = w^T x (linear combination).
    Then σ(z) gives us P(y=1|x) - the probability that class is 1!
    
    Example: z = 2 → σ(2) ≈ 0.88 → 88% confident it's class 1
             z = -3 → σ(-3) ≈ 0.05 → 5% confident (95% confident it's class 0)
    
    THE VANISHING GRADIENT PROBLEM:
    ------------------------------
    Sigmoid has a downside: for very large or very small z, the gradient becomes
    tiny (nearly zero). This makes learning slow in deep networks, which is why
    ReLU became more popular.
    
    But for the output layer of binary classification, sigmoid is still the
    standard choice!
    
    NUMERICAL STABILITY:
    -------------------
    Computing e^(-z) for large positive z can overflow!
    We use a numerically stable version:
    
    If z ≥ 0: σ(z) = 1/(1 + e^(-z))
    If z < 0: σ(z) = e^z/(1 + e^z)
    
    This avoids computing large exponentials.
    
    PARAMETERS:
    ----------
    z : float, Vector, or Matrix
        Input value(s) to transform
        
    RETURNS:
    -------
    result : same type as input
        Sigmoid-transformed value(s), all in range (0, 1)
        
    EXAMPLES:
    --------
    >>> sigmoid(0)
    0.5
    
    >>> sigmoid(2)
    0.8807970779778823  # About 88% - high confidence
    
    >>> sigmoid(-3)
    0.04742587317756678  # About 5% - low confidence
    
    >>> sigmoid(Vector([0, 2, -3]))
    Vector([0.5, 0.88, 0.05])
    """
    
    # Handle different input types
    if isinstance(z, (int, float)):
        # Single value - numerically stable computation
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        else:
            exp_z = math.exp(z)
            return exp_z / (1.0 + exp_z)
    
    elif isinstance(z, Vector):
        # Apply sigmoid to each element of vector
        result_elements = []
        for i in range(len(z)):
            val = z[i]
            if val >= 0:
                result_elements.append(1.0 / (1.0 + math.exp(-val)))
            else:
                exp_val = math.exp(val)
                result_elements.append(exp_val / (1.0 + exp_val))
        return Vector(result_elements)
    
    elif isinstance(z, Matrix):
        # Apply sigmoid to each element of matrix
        result_data = []
        for i in range(z.num_rows):
            row = []
            for j in range(z.num_cols):
                val = z[i, j]
                if val >= 0:
                    row.append(1.0 / (1.0 + math.exp(-val)))
                else:
                    exp_val = math.exp(val)
                    row.append(exp_val / (1.0 + exp_val))
            result_data.append(row)
        return Matrix(result_data)
    
    else:
        raise TypeError(f"sigmoid expects float, Vector, or Matrix, got {type(z)}")


def sigmoid_derivative(z: Union[float, Vector, Matrix]) -> Union[float, Vector, Matrix]:
    """
    Derivative of sigmoid: σ'(z) = σ(z)(1 - σ(z))
    
    This beautiful formula is used in backpropagation! Notice that the derivative
    only depends on the sigmoid value itself, which makes computation efficient.
    
    THE FORMULA EXPLAINED:
    ---------------------
    Starting from σ(z) = 1/(1+e^(-z)), using the chain rule:
    
    σ'(z) = d/dz [1/(1+e^(-z))]
          = -1/(1+e^(-z))^2 · (-e^(-z))
          = e^(-z)/(1+e^(-z))^2
          = [1/(1+e^(-z))] · [e^(-z)/(1+e^(-z))]
          = σ(z) · [1 - 1/(1+e^(-z))]
          = σ(z) · [1 - σ(z)]
    
    So we only need to compute σ(z) once, then the derivative is σ(1-σ)!
    
    THE VANISHING GRADIENT:
    ----------------------
    Notice that when σ ≈ 0 or σ ≈ 1 (when z is very negative or very positive),
    the derivative σ(1-σ) ≈ 0. This means gradients become tiny and learning slows
    down. This is the famous "vanishing gradient problem" in deep learning.
    
    PARAMETERS:
    ----------
    z : float, Vector, or Matrix
        Input value(s)
        
    RETURNS:
    -------
    derivative : same type as input
        Derivative values
    """
    sig_z = sigmoid(z)
    
    if isinstance(z, (int, float)):
        return sig_z * (1.0 - sig_z)
    elif isinstance(z, Vector):
        return Vector([sig_z[i] * (1.0 - sig_z[i]) for i in range(len(sig_z))])
    elif isinstance(z, Matrix):
        result_data = []
        for i in range(sig_z.num_rows):
            row = [sig_z[i,j] * (1.0 - sig_z[i,j]) for j in range(sig_z.num_cols)]
            result_data.append(row)
        return Matrix(result_data)


# ==============================================================================
# SOFTMAX ACTIVATION
# ==============================================================================

def softmax(z: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
    """
    Softmax activation: Converts scores to probabilities that sum to 1.
    
    This is the multi-class generalization of sigmoid! It takes a vector of scores
    (logits) and converts them to a probability distribution.
    
    THE FORMULA:
    -----------
    For input vector z = [z₁, z₂, ..., zₖ]:
    
    softmax(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)
    
    Each output is the exponential of that input divided by the sum of all exponentials.
    
    THE PROBABILITY INTERPRETATION:
    ------------------------------
    Softmax outputs a probability distribution:
    - All values between 0 and 1
    - All values sum to exactly 1
    - Larger inputs get higher probabilities
    
    Example: z = [2, 1, 0.1] (scores for 3 classes)
    softmax(z) ≈ [0.659, 0.242, 0.099]
    
    Interpretation: 66% confident it's class 0, 24% class 1, 10% class 2
    
    WHY EXPONENTIAL?
    ---------------
    1. Always positive: e^x > 0 for all x
    2. Amplifies differences: Large scores get much larger probabilities
    3. Smooth and differentiable: Enables gradient descent
    
    If we just normalized by sum (zᵢ/Σzⱼ), negative values would cause problems!
    
    THE TEMPERATURE PARAMETER:
    -------------------------
    Sometimes we use: softmax(z/T) where T is "temperature"
    - T → 0: Output becomes one-hot (argmax)
    - T = 1: Standard softmax
    - T → ∞: Output becomes uniform
    
    This controls how "confident" the predictions are.
    
    NUMERICAL STABILITY:
    -------------------
    Computing e^z for large z can overflow! We use the trick:
    
    softmax(z)ᵢ = e^(zᵢ - max(z)) / Σⱼ e^(zⱼ - max(z))
    
    Subtracting max(z) doesn't change the result (try the math!) but prevents overflow.
    
    PARAMETERS:
    ----------
    z : Vector or Matrix
        For Vector: Single sample with k class scores
        For Matrix: Multiple samples, one per row
        
    RETURNS:
    -------
    probabilities : same type as input
        Probability distribution(s) that sum to 1
        
    EXAMPLES:
    --------
    >>> z = Vector([2, 1, 0.1])
    >>> softmax(z)
    Vector([0.659, 0.242, 0.099])  # Sums to 1.0
    
    >>> # The largest score (2) gets the largest probability (66%)
    """
    
    if isinstance(z, Vector):
        # Single sample case
        
        # Numerical stability: subtract max to prevent overflow
        max_z = max(z.elements)
        exp_z = [math.exp(val - max_z) for val in z.elements]
        
        # Sum of exponentials
        sum_exp = sum(exp_z)
        
        # Normalize to get probabilities
        probabilities = [exp_val / sum_exp for exp_val in exp_z]
        
        return Vector(probabilities)
    
    elif isinstance(z, Matrix):
        # Multiple samples case - apply softmax to each row
        result_data = []
        
        for i in range(z.num_rows):
            # Get row as list
            row = [z[i, j] for j in range(z.num_cols)]
            
            # Numerical stability
            max_z = max(row)
            exp_z = [math.exp(val - max_z) for val in row]
            
            sum_exp = sum(exp_z)
            probabilities = [exp_val / sum_exp for exp_val in exp_z]
            
            result_data.append(probabilities)
        
        return Matrix(result_data)
    
    else:
        raise TypeError(f"softmax expects Vector or Matrix, got {type(z)}")


# ==============================================================================
# RELU ACTIVATION
# ==============================================================================

def relu(z: Union[float, Vector, Matrix]) -> Union[float, Vector, Matrix]:
    """
    ReLU (Rectified Linear Unit): max(0, z)
    
    This is the most popular activation in modern deep learning! It's incredibly
    simple but incredibly effective.
    
    THE FORMULA:
    -----------
    ReLU(z) = max(0, z) = { z  if z > 0
                          { 0  if z ≤ 0
    
    That's it! Just set negative values to zero.
    
    WHY IS THIS SO POWERFUL?
    -----------------------
    1. SIMPLE: Easy to compute (just a comparison)
    2. FAST: No expensive exponentials or divisions
    3. SPARSE: Produces sparse activations (many zeros)
    4. NO VANISHING GRADIENT: For positive inputs, gradient is 1
    5. NON-LINEAR: Despite being piecewise linear!
    
    THE NON-LINEARITY INSIGHT:
    -------------------------
    "Wait," I thought, "ReLU is just a linear function for positive values and zero
    for negative values. How is that non-linear?"
    
    The key: It's non-linear as a function! The composition of multiple ReLU layers
    creates very complex, non-linear transformations. This is why deep networks with
    ReLU can learn incredibly complex patterns.
    
    THE DEAD NEURON PROBLEM:
    -----------------------
    If a neuron always outputs negative values, it's "dead" - always outputs 0,
    gradient is 0, no learning happens!
    
    Solutions:
    - Initialize weights carefully
    - Use Leaky ReLU (allows small negative values)
    - Use batch normalization
    
    WHEN TO USE RELU:
    ----------------
    - Hidden layers of neural networks (default choice)
    - Whenever you want fast, effective non-linearity
    - NOT for output layer of classification (use sigmoid/softmax)
    
    PARAMETERS:
    ----------
    z : float, Vector, or Matrix
        Input value(s)
        
    RETURNS:
    -------
    result : same type as input
        max(0, z) element-wise
        
    EXAMPLES:
    --------
    >>> relu(3)
    3
    
    >>> relu(-2)
    0
    
    >>> relu(Vector([2, -1, 0, 3]))
    Vector([2, 0, 0, 3])
    """
    
    if isinstance(z, (int, float)):
        return max(0.0, float(z))
    
    elif isinstance(z, Vector):
        return Vector([max(0.0, val) for val in z.elements])
    
    elif isinstance(z, Matrix):
        result_data = []
        for i in range(z.num_rows):
            row = [max(0.0, z[i,j]) for j in range(z.num_cols)]
            result_data.append(row)
        return Matrix(result_data)
    
    else:
        raise TypeError(f"relu expects float, Vector, or Matrix, got {type(z)}")


def relu_derivative(z: Union[float, Vector, Matrix]) -> Union[float, Vector, Matrix]:
    """
    Derivative of ReLU: { 1 if z > 0
                        { 0 if z ≤ 0
    
    Super simple! The gradient is either 0 (for negative inputs) or 1 (for positive).
    
    Note: At z=0, the derivative is technically undefined, but we typically set it to 0.
    
    PARAMETERS:
    ----------
    z : float, Vector, or Matrix
        Input value(s)
        
    RETURNS:
    -------
    derivative : same type as input
        0 or 1 element-wise
    """
    
    if isinstance(z, (int, float)):
        return 1.0 if z > 0 else 0.0
    
    elif isinstance(z, Vector):
        return Vector([1.0 if val > 0 else 0.0 for val in z.elements])
    
    elif isinstance(z, Matrix):
        result_data = []
        for i in range(z.num_rows):
            row = [1.0 if z[i,j] > 0 else 0.0 for j in range(z.num_cols)]
            result_data.append(row)
        return Matrix(result_data)


# ==============================================================================
# TANH ACTIVATION
# ==============================================================================

def tanh(z: Union[float, Vector, Matrix]) -> Union[float, Vector, Matrix]:
    """
    Hyperbolic tangent: tanh(z) = (e^z - e^(-z))/(e^z + e^(-z))
    
    Similar to sigmoid, but outputs range from -1 to 1 instead of 0 to 1.
    
    WHY USE TANH INSTEAD OF SIGMOID?
    -------------------------------
    1. ZERO-CENTERED: Outputs range from -1 to 1, centered at 0
       This makes optimization easier (gradients don't all have same sign)
    
    2. STRONGER GRADIENTS: Steeper slope than sigmoid near 0
       Leads to faster convergence
    
    THE RELATIONSHIP TO SIGMOID:
    ---------------------------
    tanh(z) = 2·sigmoid(2z) - 1
    
    So tanh is just a scaled and shifted sigmoid!
    
    WHEN TO USE TANH:
    ----------------
    - Hidden layers when you want zero-centered activations
    - Recurrent neural networks (RNNs, LSTMs)
    - When your data is normalized around zero
    
    Like sigmoid, tanh suffers from vanishing gradients for large |z|.
    
    PARAMETERS:
    ----------
    z : float, Vector, or Matrix
        Input value(s)
        
    RETURNS:
    -------
    result : same type as input
        tanh-transformed values in range (-1, 1)
        
    EXAMPLES:
    --------
    >>> tanh(0)
    0.0
    
    >>> tanh(1)
    0.7615941559557649  # About 0.76
    
    >>> tanh(-1)
    -0.7615941559557649  # Symmetric around 0
    """
    
    if isinstance(z, (int, float)):
        return math.tanh(z)
    
    elif isinstance(z, Vector):
        return Vector([math.tanh(val) for val in z.elements])
    
    elif isinstance(z, Matrix):
        result_data = []
        for i in range(z.num_rows):
            row = [math.tanh(z[i,j]) for j in range(z.num_cols)]
            result_data.append(row)
        return Matrix(result_data)
    
    else:
        raise TypeError(f"tanh expects float, Vector, or Matrix, got {type(z)}")


def tanh_derivative(z: Union[float, Vector, Matrix]) -> Union[float, Vector, Matrix]:
    """
    Derivative of tanh: tanh'(z) = 1 - tanh²(z)
    
    Another beautiful formula! The derivative only depends on tanh(z) itself.
    
    PARAMETERS:
    ----------
    z : float, Vector, or Matrix
        Input value(s)
        
    RETURNS:
    -------
    derivative : same type as input
        Derivative values
    """
    tanh_z = tanh(z)
    
    if isinstance(z, (int, float)):
        return 1.0 - tanh_z ** 2
    
    elif isinstance(z, Vector):
        return Vector([1.0 - tanh_z[i] ** 2 for i in range(len(tanh_z))])
    
    elif isinstance(z, Matrix):
        result_data = []
        for i in range(tanh_z.num_rows):
            row = [1.0 - tanh_z[i,j] ** 2 for j in range(tanh_z.num_cols)]
            result_data.append(row)
        return Matrix(result_data)


# ==============================================================================
# STEP FUNCTION (for Perceptron)
# ==============================================================================

def step(z: Union[float, Vector, Matrix], threshold: float = 0.0) -> Union[float, Vector, Matrix]:
    """
    Step function (Heaviside): { 1 if z ≥ threshold
                                { 0 if z < threshold
    
    This is the original activation used in the Perceptron (1958)! It's not
    differentiable, so we can't use gradient descent with it, but it's historically
    important and still useful for binary decisions.
    
    THE PERCEPTRON'S ACTIVATION:
    ---------------------------
    The perceptron computes: z = w^T x
    Then applies: ŷ = step(z)
    
    Output is simply 0 or 1 - a hard decision, no probabilities!
    
    WHY WE MOVED AWAY FROM IT:
    --------------------------
    The step function is not differentiable at z=threshold, so we can't compute
    gradients. This is why modern models use smooth activations like sigmoid.
    
    But for simple linear separable problems, perceptron with step function
    still works perfectly!
    
    PARAMETERS:
    ----------
    z : float, Vector, or Matrix
        Input value(s)
    threshold : float
        Decision boundary (default: 0)
        
    RETURNS:
    -------
    result : same type as input
        0 or 1 element-wise
        
    EXAMPLES:
    --------
    >>> step(2)
    1
    
    >>> step(-1)
    0
    
    >>> step(Vector([2, -1, 0, 3]))
    Vector([1, 0, 1, 1])  # Note: 0 → 1 (≥ threshold)
    """
    
    if isinstance(z, (int, float)):
        return 1.0 if z >= threshold else 0.0
    
    elif isinstance(z, Vector):
        return Vector([1.0 if val >= threshold else 0.0 for val in z.elements])
    
    elif isinstance(z, Matrix):
        result_data = []
        for i in range(z.num_rows):
            row = [1.0 if z[i,j] >= threshold else 0.0 for j in range(z.num_cols)]
            result_data.append(row)
        return Matrix(result_data)
    
    else:
        raise TypeError(f"step expects float, Vector, or Matrix, got {type(z)}")


# ==============================================================================
# LEAKY RELU (Bonus - fixes dead neurons)
# ==============================================================================

def leaky_relu(z: Union[float, Vector, Matrix], alpha: float = 0.01) -> Union[float, Vector, Matrix]:
    """
    Leaky ReLU: max(alpha*z, z) where alpha is small (e.g., 0.01)
    
    This fixes the "dead neuron" problem of ReLU by allowing small negative values!
    
    THE FORMULA:
    -----------
    Leaky ReLU(z) = { z        if z > 0
                    { alpha*z  if z ≤ 0
    
    Instead of completely zeroing out negative values, we multiply them by a small
    constant (typically 0.01). This ensures gradients can flow even for negative inputs.
    
    WHY THIS HELPS:
    --------------
    With ReLU, if a neuron always outputs negative values, its gradient is always 0
    and it never learns (dead neuron).
    
    With Leaky ReLU, there's always a small gradient, so the neuron can recover!
    
    PARAMETERS:
    ----------
    z : float, Vector, or Matrix
        Input value(s)
    alpha : float
        Slope for negative values (typically 0.01)
        
    RETURNS:
    -------
    result : same type as input
        max(alpha*z, z) element-wise
    """
    
    if isinstance(z, (int, float)):
        return float(z) if z > 0 else alpha * float(z)
    
    elif isinstance(z, Vector):
        return Vector([val if val > 0 else alpha * val for val in z.elements])
    
    elif isinstance(z, Matrix):
        result_data = []
        for i in range(z.num_rows):
            row = [z[i,j] if z[i,j] > 0 else alpha * z[i,j] for j in range(z.num_cols)]
            result_data.append(row)
        return Matrix(result_data)
    
    else:
        raise TypeError(f"leaky_relu expects float, Vector, or Matrix, got {type(z)}")


# ==============================================================================
# EXAMPLE USAGE AND TESTING
# ==============================================================================

if __name__ == "__main__":
    """
    Demonstrate all activation functions with examples.
    """
    
    # Test values
    test_values = [-3, -1, 0, 1, 3]
    
    print("\n" + "=" * 70)
    print("SIGMOID: Squashes to (0, 1)")
    print("=" * 70)
    print("\n  Input  | Output  | Interpretation")
    print("---------|---------|----------------------------------")
    for val in test_values:
        sig_val = sigmoid(val)
        if sig_val > 0.5:
            interp = f"Likely class 1 ({sig_val*100:.1f}% confident)"
        else:
            interp = f"Likely class 0 ({(1-sig_val)*100:.1f}% confident)"
        print(f"  {val:5.1f}  | {sig_val:6.4f}  | {interp}")
    
    print("\n" + "=" * 70)
    print("SOFTMAX: Multi-class probabilities (sum to 1)")
    print("=" * 70)
    scores = Vector([2.0, 1.0, 0.1])
    probs = softmax(scores)
    print(f"\nClass scores: {scores.elements}")
    print(f"Probabilities: {[f'{p:.3f}' for p in probs.elements]}")
    print(f"Sum: {sum(probs.elements):.6f} (should be exactly 1.0)")
    print("\nInterpretation:")
    for i, p in enumerate(probs.elements):
        print(f"  Class {i}: {p*100:.1f}% confident")
    
    print("\n" + "=" * 70)
    print("RELU: Simple but powerful (max(0, z))")
    print("=" * 70)
    print("\n  Input  | Output  | Behavior")
    print("---------|---------|------------------")
    for val in test_values:
        relu_val = relu(val)
        behavior = "Passes through" if val > 0 else "Blocked (set to 0)"
        print(f"  {val:5.1f}  | {relu_val:6.4f}  | {behavior}")
    
    print("\n" + "=" * 70)
    print("TANH: Zero-centered (-1 to 1)")
    print("=" * 70)
    print("\n  Input  | Output  | vs Sigmoid")
    print("---------|---------|------------------")
    for val in test_values:
        tanh_val = tanh(val)
        sig_val = sigmoid(val)
        print(f"  {val:5.1f}  | {tanh_val:6.4f}  | sig={sig_val:.4f}")
    
    print("\n" + "=" * 70)
    print("COMPARISON: All activations on same inputs")
    print("=" * 70)
    test_vec = Vector([-2, -1, 0, 1, 2])
    print(f"\nInput: {test_vec.elements}")
    print(f"Sigmoid:     {[f'{v:.4f}' for v in sigmoid(test_vec).elements]}")
    print(f"Tanh:        {[f'{v:.4f}' for v in tanh(test_vec).elements]}")
    print(f"ReLU:        {[f'{v:.4f}' for v in relu(test_vec).elements]}")
    print(f"Leaky ReLU:  {[f'{v:.4f}' for v in leaky_relu(test_vec).elements]}")
    print(f"Step:        {[f'{v:.4f}' for v in step(test_vec).elements]}")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. SIGMOID: Use for binary classification output layer
   - Outputs probability between 0 and 1
   - Smooth, differentiable
   - Suffers from vanishing gradients

2. SOFTMAX: Use for multi-class classification output layer
   - Outputs probability distribution (sums to 1)
   - Generalizes sigmoid to multiple classes
   - Standard choice for classification

3. RELU: Use for hidden layers (most popular!)
   - Simple: max(0, z)
   - Fast to compute
   - No vanishing gradient for positive values
   - Can have "dead neurons" problem

4. TANH: Use for hidden layers when you want zero-centered
   - Like sigmoid but range (-1, 1)
   - Better than sigmoid for hidden layers
   - Still has vanishing gradient problem

5. STEP: Historical (perceptron)
   - Hard decision: 0 or 1
   - Not differentiable
   - Can't use gradient descent!

The choice of activation function is crucial! Wrong choice → poor performance.
Right choice → faster convergence and better results!
    """)