"""
MEAN ABSOLUTE ERROR (MAE): THE ROBUST ALTERNATIVE TO MSE
========================================================

MAE measures the average absolute difference between predictions and true values.
Unlike MSE which squares errors, MAE treats all errors linearly, making it robust
to outliers and providing different optimization properties.

TEACHING PHILOSOPHY:
-------------------
We'll explore MAE from multiple perspectives:
1. MATHEMATICAL: Linear error penalty vs quadratic
2. STATISTICAL: Median vs mean prediction, Laplacian noise
3. PRACTICAL: When MAE beats MSE and vice versa
4. OPTIMIZATION: Non-smooth gradient and its challenges

THE MATHEMATICAL DEFINITION:
---------------------------
For predictions ŷ and true values y:

MAE = (1/n) Σ|ŷᵢ - yᵢ|

Where:
- n is the number of samples
- |·| is the absolute value (always positive)
- We sum absolute errors and average

Compare to MSE = (1/n) Σ(ŷᵢ - yᵢ)²

THE KEY DIFFERENCE: LINEAR VS QUADRATIC
---------------------------------------

**MAE (Linear penalty):**
- Error of 1 → contributes 1 to loss
- Error of 10 → contributes 10 to loss
- Error of 100 → contributes 100 to loss
Penalty grows LINEARLY with error size.

**MSE (Quadratic penalty):**
- Error of 1 → contributes 1 to loss
- Error of 10 → contributes 100 to loss
- Error of 100 → contributes 10,000 to loss
Penalty grows QUADRATICALLY with error size.

This fundamental difference makes MAE robust to outliers!

Example:
Errors: [0.5, 0.8, 1.2, 50.0]  (one huge outlier)

MAE = (0.5 + 0.8 + 1.2 + 50.0) / 4 = 13.125
MSE = (0.25 + 0.64 + 1.44 + 2500) / 4 = 625.6

The outlier dominates MSE but contributes proportionally to MAE!

WHY ABSOLUTE VALUES?
-------------------

**1. ERRORS MUST BE POSITIVE**
Without absolute value, errors cancel: +10 and -10 average to 0.
Taking absolute value ensures all errors count.

**2. TREATS ALL ERRORS EQUALLY**
A 10-unit error contributes 10 to the loss, whether it's
over-prediction or under-prediction. Democratic penalty!

**3. CORRESPONDS TO L1 NORM**
MAE is the L1 distance (Manhattan distance) between vectors.
This has useful properties for regularization and optimization.

STATISTICAL INTERPRETATION: MEDIAN PREDICTION
---------------------------------------------
Here's something profound:

**MSE optimal predictor: MEAN**
If you minimize MSE, the best constant prediction is the mean of y.

**MAE optimal predictor: MEDIAN**
If you minimize MAE, the best constant prediction is the median of y!

Why this matters:
- Median is robust to outliers (50th percentile, unaffected by extremes)
- Mean is sensitive to outliers (one huge value shifts the mean)
- MAE leads to robust predictions, MSE to mean predictions

Example:
Data: [1, 2, 3, 4, 100]  (outlier at 100)
Mean = 22 (pulled up by outlier)
Median = 3 (unaffected by outlier)

MAE optimal prediction: 3 (robust)
MSE optimal prediction: 22 (sensitive)

LAPLACIAN NOISE ASSUMPTION:
---------------------------
Just as MSE assumes Gaussian noise, MAE assumes Laplacian noise!

**Laplacian distribution:**
- Sharper peak than Gaussian (more data near center)
- Heavier tails than Gaussian (more outliers)
- Corresponds to L1 norm

If your errors follow a Laplacian distribution, MAE is the
maximum likelihood estimator. If errors are Gaussian, use MSE.

THE GRADIENT CHALLENGE:
----------------------
MAE gradient: ∂MAE/∂ŷ = (1/n) Σ sign(ŷᵢ - yᵢ)

Where sign(x) = +1 if x > 0, -1 if x < 0, 0 if x = 0

**The problem:** sign function is not differentiable at 0!

At x = 0, the gradient has a discontinuity:
- Approaching from left: gradient = -1
- Approaching from right: gradient = +1
- At exactly 0: gradient undefined!

This creates challenges for gradient-based optimization:
- Gradient descent can oscillate near the minimum
- Convergence is slower than MSE
- Need careful learning rate tuning

**Solutions:**
1. Use subgradient (any value in [-1, +1] at zero)
2. Smooth approximation: Huber loss
3. Smaller learning rates

THE GRADIENT MAGNITUDE:
----------------------
Another key difference from MSE:

**MAE gradient:** Constant magnitude (±1/n)
- Large error: gradient = +1/n or -1/n
- Small error: gradient = +1/n or -1/n
- Same magnitude regardless of error size!

**MSE gradient:** Proportional to error
- Large error: large gradient
- Small error: small gradient
- Gradient magnitude = (2/n) × error

MAE implications:
- Updates have constant size (only direction changes)
- Can overshoot near convergence (no automatic slowing)
- Need learning rate decay for precise convergence

MAE VS MSE: WHEN TO USE WHICH
-----------------------------

**Use MAE when:**
✓ Data has outliers that are measurement errors (should be downweighted)
✓ You want robust predictions (median-like behavior)
✓ Errors follow Laplacian distribution
✓ Under/over-prediction equally costly
✓ You can tolerate slower convergence

**Use MSE when:**
✓ Data has no outliers, or outliers are meaningful
✓ You want mean predictions (average-like behavior)
✓ Errors follow Gaussian distribution
✓ Large errors are particularly bad (quadratic penalty needed)
✓ You need fast, smooth convergence

**Use Huber Loss when:**
✓ You want the best of both worlds
✓ MSE for small errors (smooth convergence)
✓ MAE for large errors (robust to outliers)

HUBER LOSS: THE COMPROMISE
--------------------------
Huber loss combines MSE and MAE:

Huber(δ) = { ½(ŷ - y)²           if |ŷ - y| ≤ δ
           { δ|ŷ - y| - ½δ²     if |ŷ - y| > δ

Where δ is a threshold:
- Small errors (< δ): Use MSE (smooth, fast convergence)
- Large errors (> δ): Use MAE (robust to outliers)

Best of both worlds! Used in many production systems.

QUANTILE REGRESSION:
-------------------
MAE generalizes to quantile regression!

Instead of predicting the median (50th percentile), you can
predict any quantile by using asymmetric absolute error:

Loss = (1/n) Σ ρ_τ(ŷᵢ - yᵢ)

Where ρ_τ(u) = u(τ - 𝟙(u < 0))

For τ = 0.5: reduces to MAE (median)
For τ = 0.9: predicts 90th percentile
For τ = 0.1: predicts 10th percentile

This lets you model prediction intervals, not just point estimates!

MACHINE LEARNING APPLICATIONS:
-----------------------------

**1. ROBUST REGRESSION**
When data has outliers from measurement errors or rare events.
MAE prevents outliers from dominating the fit.

**2. TIME SERIES FORECASTING**
Stock prices, weather: outliers are common (crashes, storms).
MAE provides more stable predictions than MSE.

**3. REINFORCEMENT LEARNING**
In some RL applications, you want median return not mean return.
Using MAE in value function learning achieves this.

**4. COMPUTER VISION**
Object detection, tracking: outliers from occlusions, motion blur.
L1 loss more robust than L2.

**5. ANOMALY DETECTION**
When you specifically want to downweight anomalies.
MAE treats them as just another error, not catastrophic.

Let's implement MAE with all its properties!
"""


from typing import Any, List

from foundations.linear_algebra.vectors_and_matrices import Matrix

from .mse_loss import mean_squared_error


def mean_absolute_error(y_true: Matrix, y_pred: Matrix) -> float:
    """
    Compute the mean absolute error between true and predicted values using Matrix.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"y_pred has shape {y_pred.shape}"
        )
    
    # Calculate (y_pred - y_true)
    errors_matrix = y_pred - y_true
    
    # Apply absolute value to each error element-wise
    absolute_errors_matrix = errors_matrix.apply_func(abs)
    
    # Average over all elements
    mae = absolute_errors_matrix.mean_all()
    
    return float(mae)


def median_absolute_error(y_true: Matrix, y_pred: Matrix) -> float:
    """
    Compute the median absolute error (MedAE) using Matrix.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"y_pred has shape {y_pred.shape}"
        )

    # Compute absolute errors
    errors_matrix = y_pred - y_true
    absolute_errors_list = errors_matrix.apply_func(abs).flatten()
    
    if not absolute_errors_list:
        return 0.0 # Handle empty list case
        
    # Sort the list to find the median
    absolute_errors_list.sort()
    n = len(absolute_errors_list)
    
    if n % 2 == 1:
        # Odd number of elements, median is the middle one
        medae = absolute_errors_list[n // 2]
    else:
        # Even number of elements, median is average of the two middle ones
        mid1 = absolute_errors_list[n // 2 - 1]
        mid2 = absolute_errors_list[n // 2]
        medae = (mid1 + mid2) / 2.0
    
    return float(medae)


def mae_gradient(y_true: Matrix, y_pred: Matrix) -> Matrix:
    """
    Compute the gradient of MAE with respect to predictions, using Matrix.
    
    ∂MAE/∂y_pred = (1/n_elements) * sign(y_pred - y_true)
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"y_pred has shape {y_pred.shape}"
        )
    
    n_elements = y_true.num_rows * y_true.num_cols
    if n_elements == 0:
        return Matrix([[]]) # Return empty gradient for empty input
    
    # Calculate error
    error_matrix = y_pred - y_true
    
    # Define a custom sign function for numbers
    def sign_func(x):
        if x > 0:
            return 1.0
        elif x < 0:
            return -1.0
        else:
            return 0.0 # Subgradient choice for x=0
            
    # Apply sign function element-wise
    signed_errors_matrix = error_matrix.apply_func(sign_func)
    
    # Scale by (1/n_elements)
    gradient_matrix = signed_errors_matrix / n_elements
    
    return gradient_matrix


def huber_loss(y_true: Matrix,
               y_pred: Matrix,
               delta: float = 1.0) -> float:
    """
    Compute Huber loss: combines MSE and MAE for best of both worlds, using Matrix.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"y_pred has shape {y_pred.shape}"
        )
    
    # Compute errors
    error_matrix = y_pred - y_true
    abs_error_matrix = error_matrix.apply_func(abs)
    
    total_loss = 0.0
    num_elements = y_true.num_rows * y_true.num_cols
    
    if num_elements == 0:
        return 0.0

    for r in range(y_true.num_rows):
        for c in range(y_true.num_cols):
            err_val = error_matrix.elements[r][c]
            abs_err_val = abs_error_matrix.elements[r][c]

            if abs_err_val <= delta:
                # MSE region
                total_loss += 0.5 * err_val ** 2
            else:
                # MAE-like region
                total_loss += delta * abs_err_val - 0.5 * delta ** 2
                
    return total_loss / num_elements


# ==============================================================================
# USAGE EXAMPLES AND TESTS
# ==============================================================================

# Helper function to generate Matrix data for examples
def create_matrix(data: List[List[Any]]) -> Matrix:
    # Ensure inner lists are handled for single-row inputs as well
    if not data or not isinstance(data[0], list):
        return Matrix([data])
    return Matrix(data)

if __name__ == "__main__":
    print("=" * 70)
    print("MEAN ABSOLUTE ERROR (MAE): INTERACTIVE TUTORIAL (using custom Matrix)")
    print("=" * 70)
    
    print("\n### BASIC MAE COMPUTATION ###\n")
    
    y_true_data = [[3.0], [-0.5], [2.0], [7.0]]
    y_pred_data = [[2.5], [0.0], [2.0], [8.0]]
    
    y_true = create_matrix(y_true_data)
    y_pred = create_matrix(y_pred_data)
    
    print("True values:", y_true)
    print("Predictions:", y_pred)
    
    errors_display = y_pred - y_true
    print("Errors (pred - true):", errors_display)
    
    abs_errors_display = errors_display.apply_func(abs)
    print("Absolute errors:", abs_errors_display)
    
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\nMAE = {mae:.4f}")
    print(f"  → Average prediction is off by {mae:.4f} units")
    
    print("\n### MAE VS MSE: THE OUTLIER SENSITIVITY TEST ###\n")
    
    # Data without outliers
    y_normal = create_matrix([[1], [2], [3], [4], [5]])
    y_pred_normal = create_matrix([[1.1], [2.1], [2.9], [4.1], [4.9]])
    
    # Data with one outlier
    y_with_outlier = create_matrix([[1], [2], [3], [4], [100]])
    y_pred_outlier = create_matrix([[1.1], [2.1], [2.9], [4.1], [50]])
    
    print("Normal data (no outliers):")
    mae_normal = mean_absolute_error(y_normal, y_pred_normal)
    mse_normal = mean_squared_error(y_normal, y_pred_normal)
    print(f"  MAE: {mae_normal:.4f}")
    print(f"  MSE: {mse_normal:.4f}")
    
    print("\nData with outlier (true=100, pred=50):")
    mae_outlier = mean_absolute_error(y_with_outlier, y_pred_outlier)
    mse_outlier = mean_squared_error(y_with_outlier, y_pred_outlier)
    print(f"  MAE: {mae_outlier:.4f}")
    print(f"  MSE: {mse_outlier:.4f}")
    
    print(f"\nImpact of outlier:")
    # Ensure division by zero is handled if normal MSE is 0
    mae_ratio = mae_outlier / mae_normal if mae_normal != 0 else float('inf')
    mse_ratio = mse_outlier / mse_normal if mse_normal != 0 else float('inf')

    print(f"  MAE increased by: {mae_ratio:.1f}x")
    print(f"  MSE increased by: {mse_ratio:.1f}x")
    print("\n→ MSE is FAR more sensitive to outliers!")
    
    print("\n### MAE GRADIENT: CONSTANT MAGNITUDE ###\n")
    
    y_true_grad = create_matrix([[0.0], [0.0], [0.0], [0.0]])
    y_pred_small = create_matrix([[0.1], [0.0], [-0.1], [0.0]])
    y_pred_large = create_matrix([[10.0], [0.0], [-10.0], [0.0]])
    
    grad_small = mae_gradient(y_true_grad, y_pred_small)
    grad_large = mae_gradient(y_true_grad, y_pred_large)
    
    print("Small errors (0.1, 0, -0.1, 0):")
    print(f"  Gradient: {grad_small}")
    
    print("\nLarge errors (10, 0, -10, 0):")
    print(f"  Gradient: {grad_large}")
    
    print("\n→ Same gradient magnitude regardless of error size!")
    print("  This is why MAE is robust to outliers")
    print("  But also why convergence is slower (no automatic slowing)")
    
    print("\n### MEDIAN PREDICTION PROPERTY ###\n")
    
    # Data with outlier
    y_data_list = [[1], [2], [3], [4], [100]]
    y_data = create_matrix(y_data_list)
    
    y_data_flat = y_data.flatten()
    mean_val = sum(y_data_flat) / len(y_data_flat)
    
    # Calculate median manually for pure Python context
    sorted_y_data_flat = sorted(y_data_flat)
    n_y_data = len(sorted_y_data_flat)
    if n_y_data % 2 == 1:
        median_val = sorted_y_data_flat[n_y_data // 2]
    else:
        median_val = (sorted_y_data_flat[n_y_data // 2 - 1] + sorted_y_data_flat[n_y_data // 2]) / 2.0
    
    # Predict mean for all samples
    y_pred_mean_matrix = create_matrix([[mean_val] for _ in y_data_list])
    mae_mean = mean_absolute_error(y_data, y_pred_mean_matrix)
    mse_mean = mean_squared_error(y_data, y_pred_mean_matrix)
    
    # Predict median for all samples
    y_pred_median_matrix = create_matrix([[median_val] for _ in y_data_list])
    mae_median = mean_absolute_error(y_data, y_pred_median_matrix)
    mse_median = mean_squared_error(y_data, y_pred_median_matrix)
    
    print(f"Data: {y_data}")
    print(f"Mean: {mean_val:.1f}")
    print(f"Median: {median_val:.1f}")
    
    print(f"\nAlways predict mean ({mean_val:.1f}):")
    print(f"  MAE: {mae_mean:.4f}")
    print(f"  MSE: {mse_mean:.4f} ← MSE is minimized")
    
    print(f"\nAlways predict median ({median_val:.1f}):")
    print(f"  MAE: {mae_median:.4f} ← MAE is minimized")
    print(f"  MSE: {mse_median:.4f}")
    
    print("\n→ MAE favors median (robust), MSE favors mean (sensitive)")
    
    print("\n### HUBER LOSS: BEST OF BOTH WORLDS ###\n")
    
    y_true_huber_data = [[1], [2], [3], [4], [5]]
    y_pred_good_data = [[1.1], [2.1], [2.9], [4.1], [4.9]]
    y_pred_outlier_data = [[1.1], [2.1], [2.9], [4.1], [50]]
    
    y_true_huber = create_matrix(y_true_huber_data)
    y_pred_good = create_matrix(y_pred_good_data)
    y_pred_outlier = create_matrix(y_pred_outlier_data)
    
    print("Good predictions (small errors):")
    mae_good = mean_absolute_error(y_true_huber, y_pred_good)
    mse_good = mean_squared_error(y_true_huber, y_pred_good)
    huber_good = huber_loss(y_true_huber, y_pred_good, delta=1.0)
    print(f"  MAE: {mae_good:.4f}")
    print(f"  MSE: {mse_good:.4f}")
    print(f"  Huber: {huber_good:.4f}")
    print("  → All losses similar (no outliers)")
    
    print("\nWith outlier (last prediction is 50 instead of 5):")
    mae_outlier_huber = mean_absolute_error(y_true_huber, y_pred_outlier)
    mse_outlier_huber = mean_squared_error(y_true_huber, y_pred_outlier)
    huber_outlier_huber = huber_loss(y_true_huber, y_pred_outlier, delta=1.0)
    print(f"  MAE: {mae_outlier_huber:.4f} (robust)")
    print(f"  MSE: {mse_outlier_huber:.4f} (dominated by outlier!)")
    print(f"  Huber: {huber_outlier_huber:.4f} (balanced)")
    print("  → Huber is between MAE and MSE")
    
    print("\n### MEDIAN ABSOLUTE ERROR (MedAE) ###\n")
    
    y_true_medae_data = [[1], [2], [3], [4], [5]]
    y_pred_medae_data = [[1.1], [2.2], [2.9], [4.3], [100]]
    
    y_true_medae = create_matrix(y_true_medae_data)
    y_pred_medae = create_matrix(y_pred_medae_data)
    
    abs_errors_medae_display = (y_pred_medae - y_true_medae).apply_func(abs)
    print(f"Absolute errors: {abs_errors_medae_display.flatten()}")
    
    mae_val_medae = mean_absolute_error(y_true_medae, y_pred_medae)
    medae_val_medae = median_absolute_error(y_true_medae, y_pred_medae)
    
    print(f"MAE (mean of errors): {mae_val_medae:.4f}")
    print(f"MedAE (median of errors): {medae_val_medae:.4f}")
    print("\n→ MedAE shows typical error (0.3) despite huge outlier!")
    print("  Use for reporting when data has outliers")
    
    print("\n### KEY TAKEAWAYS ###\n")
    print("""
1. MAE = average absolute error (linear penalty)
   - Robust to outliers (constant gradient)
   - Optimal predictor is median (not mean)
   - Slower convergence (need learning rate decay)

2. MAE vs MSE:
   - Use MAE: outliers, robust predictions, Laplacian noise
   - Use MSE: no outliers, fast convergence, Gaussian noise
   - Use Huber: best of both worlds!

3. Gradient has constant magnitude ±1/n
   - Large and small errors update equally
   - Prevents outliers from dominating updates
   - Requires careful learning rate tuning

4. MedAE (median absolute error):
   - Even more robust than MAE
   - Best for reporting typical error
   - Not useful for optimization
    """)
    
    print("\n✅ MAE loss (with custom Matrix) complete!")