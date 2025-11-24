"""
LINEAR REGRESSION VISUALIZATIONS: Seeing Is Believing!
======================================================

This file creates visualizations to help understand what linear regression is actually doing.
I didn't truly "get" gradient descent until I saw it in action — watching the loss decrease
and the line fit the data made everything click!

WHAT WE'LL VISUALIZE:
--------------------
1. Simple 2D regression: Plot data points and the fitted line
2. Loss convergence: Watch loss decrease over epochs
3. Learning rate comparison: See how different learning rates affect convergence
4. Polynomial regression: Visualize fitting curves with different degrees
5. Optimizer comparison: Compare Batch, SGD, Mini-Batch side by side
6. Residuals: Show prediction errors visually

IMPORTANT NOTE:
--------------
We're using pure Python for plotting (no matplotlib!). I've implemented a simple
ASCII-based plotting system that works in the terminal. It's not as pretty as
matplotlib, but it's cool and shows you don't always need external libraries!
"""


from foundations.linear_algebra.vectors_and_matrices import Matrix
from linear_models.linear_regression.linear_regression_scratch import LinearRegression
from linear_models.linear_regression.linear_regression_closed_form import LinearRegressionClosedForm
from linear_models.linear_regression.polynomial_features import PolynomialFeatures
from typing import List
import random


# ==============================================================================
# ASCII PLOTTING UTILITIES
# ==============================================================================

def plot_2d_regression(
    X: Matrix, 
    y: Matrix, 
    model: LinearRegression,
    title: str = "Linear Regression Fit",
    width: int = 60,
    height: int = 20
):
    """
    Create an ASCII plot showing data points and the regression line.
    
    This visualizes the classic 2D regression problem: given (x, y) points,
    plot the fitted line y = w₀ + w₁x.
    
    THE VISUALIZATION:
    -----------------
    - Data points shown as 'o'
    - Regression line shown as '*'
    - Axes labeled with values
    
    PARAMETERS:
    ----------
    X : Matrix, shape (n, 2)
        Features [bias, x]
    y : Matrix, shape (n, 1)
        Targets
    model : LinearRegression
        Fitted model
    title : str
        Plot title
    width : int
        Plot width in characters
    height : int
        Plot height in characters
    """
    
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)
    
    # Extract x values (skip bias column)
    x_values = [X[i, 1] for i in range(X.num_rows)]
    y_values = [y[i, 0] for i in range(y.num_rows)]
    
    # Get predictions for the line
    y_pred_values = []
    for i in range(X.num_rows):
        pred = model.predict(Matrix([X.elements[i]]))
        y_pred_values.append(pred[0, 0])
    
    # Find ranges
    x_min, x_max = min(x_values), max(x_values)
    y_min_data, y_max_data = min(y_values), max(y_values)
    y_min_pred, y_max_pred = min(y_pred_values), max(y_pred_values)
    
    y_min = min(y_min_data, y_min_pred)
    y_max = max(y_max_data, y_max_pred)
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.1
    x_max += x_range * 0.1
    y_min -= y_range * 0.1
    y_max += y_range * 0.1
    
    # Create empty plot
    plot = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Helper function to convert data coordinates to plot coordinates
    def to_plot_coords(x, y):
        plot_x = int((x - x_min) / (x_max - x_min) * (width - 1))
        plot_y = height - 1 - int((y - y_min) / (y_max - y_min) * (height - 1))
        return plot_x, plot_y
    
    # Plot the regression line
    x_line_values = [x_min + (x_max - x_min) * i / (width - 1) for i in range(width)]
    for x_line in x_line_values:
        # Predict y for this x
        X_line = Matrix([[1, x_line]])
        y_line = model.predict(X_line)[0, 0]
        
        px, py = to_plot_coords(x_line, y_line)
        if 0 <= px < width and 0 <= py < height:
            plot[py][px] = '*'
    
    # Plot data points (they should overwrite the line)
    for x_data, y_data in zip(x_values, y_values):
        px, py = to_plot_coords(x_data, y_data)
        if 0 <= px < width and 0 <= py < height:
            plot[py][px] = 'o'
    
    # Print the plot
    print(f"\n{y_max:7.2f} |", end='')
    for row in plot:
        print(''.join(row))
        if row != plot[-1]:
            print("        |", end='')
    
    print(f"{y_min:7.2f} |")
    print("        " + "-" * width)
    print(f"        {x_min:7.2f}" + " " * (width - 16) + f"{x_max:7.2f}")
    
    # Print model equation
    w0, w1 = model.weights_[0], model.weights_[1]
    print(f"\nEquation: y = {w0:.4f} + {w1:.4f}·x")
    print(f"R² score: {model.score(X, y):.4f}")
    
    print("Legend: o = data points, * = regression line")


def plot_loss_curve(
    history: dict,
    title: str = "Loss vs Epoch",
    width: int = 60,
    height: int = 15
):
    """
    Plot the loss curve showing how loss decreases over epochs.
    
    This is one of the most important visualizations! It shows you whether:
    - Training is working (loss decreasing)
    - Learning rate is appropriate (smooth decrease vs oscillation)
    - Model has converged (loss plateaued)
    
    PARAMETERS:
    ----------
    history : dict
        Training history with 'loss' and 'epoch' keys
    title : str
        Plot title
    width : int
        Plot width
    height : int
        Plot height
    """
    
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)
    
    losses = history['loss']
    epochs = history['epoch']
    
    if not losses:
        print("No training history to plot!")
        return
    
    # Find range
    loss_min, loss_max = min(losses), max(losses)
    epoch_min, epoch_max = min(epochs), max(epochs)
    
    # Add padding
    loss_range = loss_max - loss_min
    if loss_range < 1e-10:
        loss_range = 1.0
    loss_min -= loss_range * 0.1
    loss_max += loss_range * 0.1
    
    # Create empty plot
    plot = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Plot the loss curve
    for epoch, loss in zip(epochs, losses):
        # Convert to plot coordinates
        px = int((epoch - epoch_min) / (epoch_max - epoch_min + 1) * (width - 1))
        py = height - 1 - int((loss - loss_min) / (loss_max - loss_min) * (height - 1))
        
        if 0 <= px < width and 0 <= py < height:
            plot[py][px] = '*'
    
    # Print the plot
    print(f"\n{loss_max:9.4f} |", end='')
    for row in plot:
        print(''.join(row))
        if row != plot[-1]:
            print("          |", end='')
    
    print(f"{loss_min:9.4f} |")
    print("          " + "-" * width)
    print(f"          Epoch 0" + " " * (width - 20) + f"Epoch {epoch_max}")
    
    # Print statistics
    print(f"\nInitial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Reduction: {(1 - losses[-1]/losses[0])*100:.2f}%")
    
    # Convergence assessment
    if len(losses) > 10:
        recent_change = abs(losses[-1] - losses[-5]) / losses[-5] if losses[-5] > 0 else 0
        if recent_change < 0.01:
            print("\n✓ Converged! (loss change < 1% in last few epochs)")
        else:
            print(f"\n→ Still improving (recent change: {recent_change*100:.2f}%)")


def compare_learning_rates(
    X: Matrix,
    y: Matrix,
    learning_rates: List[float] = [0.001, 0.01, 0.1, 0.5]
):
    """
    Compare convergence with different learning rates.
    
    This demonstrates one of the most important hyperparameters in gradient descent!
    
    WHAT YOU'LL SEE:
    ---------------
    - Too small: Slow convergence, takes forever
    - Just right: Smooth, steady decrease
    - Too large: Oscillation or divergence
    
    PARAMETERS:
    ----------
    X : Matrix
        Training features
    y : Matrix
        Training targets
    learning_rates : List[float]
        Learning rates to compare
    """
    
    print("\n" + "=" * 70)
    print("LEARNING RATE COMPARISON")
    print("=" * 70)
    print("\nTraining the same model with different learning rates...")
    print("Watch how learning rate affects convergence!")
    print()
    
    for lr in learning_rates:
        print(f"\nLearning rate: {lr}")
        print("-" * 40)
        
        try:
            model = LinearRegression(
                learning_rate=lr,
                n_epochs=50,
                verbose=False
            )
            model.fit(X, y)
            
            losses = model.history_['loss']
            initial_loss = losses[0]
            final_loss = losses[-1]
            
            # Check for divergence
            if final_loss > initial_loss * 2:
                print(f"  ✗ DIVERGED! Loss increased from {initial_loss:.4f} to {final_loss:.4f}")
                print(f"    Learning rate is too large!")
            elif final_loss > initial_loss * 0.99:
                print(f"  → Slow convergence: {initial_loss:.4f} → {final_loss:.4f}")
                print(f"    Reduction: {(1 - final_loss/initial_loss)*100:.2f}%")
                print(f"    Learning rate might be too small")
            else:
                print(f"  ✓ Good convergence: {initial_loss:.4f} → {final_loss:.4f}")
                print(f"    Reduction: {(1 - final_loss/initial_loss)*100:.2f}%")
            
            # Show simple loss trajectory
            print("  Loss trajectory:", end='')
            step = max(1, len(losses) // 10)
            for i in range(0, len(losses), step):
                print(f" {losses[i]:.2f}", end='')
            print()
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}")


def compare_polynomial_degrees(
    X: Matrix,
    y: Matrix,
    degrees: List[int] = [1, 2, 3, 5]
):
    """
    Compare regression with different polynomial degrees.
    
    This shows the progression from underfitting to overfitting!
    
    WHAT YOU'LL SEE:
    ---------------
    - Degree 1: Straight line, might underfit
    - Degree 2-3: Smooth curve, usually just right
    - High degree: Wiggly curve, overfits training data
    
    PARAMETERS:
    ----------
    X : Matrix
        Training features (single feature, no bias)
    y : Matrix
        Training targets
    degrees : List[int]
        Polynomial degrees to try
    """
    
    print("\n" + "=" * 70)
    print("POLYNOMIAL DEGREE COMPARISON")
    print("=" * 70)
    print("\nFitting polynomials of different degrees to the same data...")
    print("Watch how model complexity affects the fit!")
    print()
    
    for degree in degrees:
        print(f"\nDegree {degree}:")
        print("-" * 40)
        
        # Transform features
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly = poly.fit_transform(X)
        
        print(f"  Features: {poly.get_feature_names()}")
        print(f"  Number of features: {X.num_cols} → {X_poly.num_cols}")
        
        # Fit model
        model = LinearRegression(
            learning_rate=0.001 if degree > 2 else 0.01,
            n_epochs=500,
            verbose=False
        )
        model.fit(X_poly, y)
        
        # Evaluate
        r2 = model.score(X_poly, y)
        print(f"  R² score: {r2:.4f}")
        
        # Interpretation
        if r2 > 0.99:
            print(f"  → Nearly perfect fit! Might be overfitting...")
        elif r2 > 0.8:
            print(f"  → Good fit!")
        else:
            print(f"  → Poor fit, might be underfitting")


def compare_optimizers(
    X: Matrix,
    y: Matrix
):
    """
    Compare Batch, SGD, and Mini-Batch gradient descent.
    
    This shows the fundamental tradeoff between convergence speed and stability!
    
    WHAT YOU'LL SEE:
    ---------------
    - Batch GD: Smooth curve, slow
    - SGD: Noisy curve, fast initially
    - Mini-Batch: Middle ground
    
    PARAMETERS:
    ----------
    X : Matrix
        Training features
    y : Matrix
        Training targets
    """
    
    print("\n" + "=" * 70)
    print("OPTIMIZER COMPARISON")
    print("=" * 70)
    print("\nComparing Batch, SGD, and Mini-Batch gradient descent...")
    print()
    
    optimizers = {
        'Batch GD': ('batch', 0.01),
        'SGD': ('sgd', 0.005),
        'Mini-Batch GD': ('mini_batch', 0.01)
    }
    
    for name, (opt_type, lr) in optimizers.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        model = LinearRegression(
            optimizer=opt_type,
            learning_rate=lr,
            n_epochs=50,
            batch_size=8,
            verbose=False
        )
        
        model.fit(X, y)
        
        losses = model.history_['loss']
        if losses:
            print(f"  Initial loss: {losses[0]:.6f}")
            print(f"  Final loss: {losses[-1]:.6f}")
            print(f"  Reduction: {(1 - losses[-1]/losses[0])*100:.2f}%")
            
            # Assess convergence quality
            if opt_type == 'batch':
                print(f"  Characteristics: Smooth, deterministic")
            elif opt_type == 'sgd':
                print(f"  Characteristics: Noisy, explores more")
            else:
                print(f"  Characteristics: Balanced, practical")


# ==============================================================================
# MAIN DEMONSTRATION
# ==============================================================================

if __name__ == "__main__":
    """
    Run all visualizations to demonstrate linear regression concepts.
    """
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║       LINEAR REGRESSION VISUALIZATIONS: SEEING IS BELIEVING     ║
║                                                                  ║
║  "I didn't truly understand gradient descent until I watched   ║
║   it learn in real-time!" - Me, after building these plots      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # =========================================================================
    # Demo 1: Simple 2D Regression
    # =========================================================================
    
    print("\n" + "#" * 70)
    print("DEMO 1: SIMPLE 2D LINEAR REGRESSION")
    print("#" * 70)
    
    # Create simple linear data
    X_simple = Matrix([
        [1, 1], [1, 2], [1, 3], [1, 4], [1, 5],
        [1, 6], [1, 7], [1, 8], [1, 9], [1, 10]
    ])
    y_simple = Matrix([[3], [5], [7], [9], [11], [13], [15], [17], [19], [21]])
    
    print("\nData: y = 1 + 2x (perfect linear relationship)")
    
    model_simple = LinearRegression(learning_rate=0.01, n_epochs=100, verbose=False)
    model_simple.fit(X_simple, y_simple)
    
    plot_2d_regression(X_simple, y_simple, model_simple, 
                      title="Perfect Linear Fit (y = 1 + 2x)")
    
    plot_loss_curve(model_simple.history_, title="Loss Convergence")
    
    # =========================================================================
    # Demo 2: Noisy Data
    # =========================================================================
    
    print("\n" + "#" * 70)
    print("DEMO 2: REGRESSION WITH NOISY DATA")
    print("#" * 70)
    
    # Create noisy data
    random.seed(42)
    X_noisy_data = [[1, float(i)] for i in range(1, 11)]
    y_noisy_data = [[float(i * 2 + 1 + random.gauss(0, 1.5))] for i in range(1, 11)]
    
    X_noisy = Matrix(X_noisy_data)
    y_noisy = Matrix(y_noisy_data)
    
    print("\nData: y = 1 + 2x + noise")
    
    model_noisy = LinearRegression(learning_rate=0.01, n_epochs=200, verbose=False)
    model_noisy.fit(X_noisy, y_noisy)
    
    plot_2d_regression(X_noisy, y_noisy, model_noisy,
                      title="Regression with Noisy Data")
    
    # =========================================================================
    # Demo 3: Learning Rate Comparison
    # =========================================================================
    
    print("\n" + "#" * 70)
    print("DEMO 3: EFFECT OF LEARNING RATE")
    print("#" * 70)
    
    compare_learning_rates(X_simple, y_simple, 
                          learning_rates=[0.001, 0.01, 0.1, 0.5, 1.0])
    
    # =========================================================================
    # Demo 4: Polynomial Regression
    # =========================================================================
    
    print("\n" + "#" * 70)
    print("DEMO 4: POLYNOMIAL REGRESSION")
    print("#" * 70)
    
    # Create parabolic data
    X_para_data = [[float(i)] for i in range(1, 8)]
    y_para_data = [[float(i**2)] for i in range(1, 8)]
    
    X_para = Matrix(X_para_data)
    y_para = Matrix(y_para_data)
    
    print("\nData: y = x² (parabola)")
    
    compare_polynomial_degrees(X_para, y_para, degrees=[1, 2, 3, 5])
    
    # =========================================================================
    # Demo 5: Optimizer Comparison
    # =========================================================================
    
    print("\n" + "#" * 70)
    print("DEMO 5: COMPARING OPTIMIZERS")
    print("#" * 70)
    
    compare_optimizers(X_noisy, y_noisy)
    
    # =========================================================================
    # Demo 6: Closed-Form vs Gradient Descent
    # =========================================================================
    
    print("\n" + "#" * 70)
    print("DEMO 6: CLOSED-FORM VS GRADIENT DESCENT")
    print("#" * 70)
    
    print("\nBoth methods should give the same result...")
    print()
    
    model_gd = LinearRegression(learning_rate=0.01, n_epochs=200, verbose=False)
    model_gd.fit(X_simple, y_simple)
    
    model_cf = LinearRegressionClosedForm()
    model_cf.fit(X_simple, y_simple)
    
    print("Gradient Descent:")
    print(f"  Weights: {[f'{w:.6f}' for w in model_gd.weights_.elements]}")
    print(f"  R² score: {model_gd.score(X_simple, y_simple):.6f}")
    
    print("\nClosed-Form (Normal Equation):")
    print(f"  Weights: {[f'{w:.6f}' for w in model_cf.weights_.elements]}")
    print(f"  R² score: {model_cf.score(X_simple, y_simple):.6f}")
    
    print("\n✓ Both methods converge to the same solution!")
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("VISUALIZATION SUMMARY")
    print("=" * 70)
    print("""
What we learned by visualizing:

1. REGRESSION PLOTS show how well the line fits the data
   - Points close to line = good fit
   - Points far from line = high error

2. LOSS CURVES show the learning process
   - Smooth decrease = good learning rate
   - Oscillating = learning rate too large
   - Flat = converged or learning rate too small

3. LEARNING RATE is critical!
   - Too small: Slow convergence (takes forever)
   - Just right: Smooth, steady decrease
   - Too large: Divergence (loss increases!)

4. POLYNOMIAL FEATURES allow fitting curves
   - Degree 1: Straight line
   - Degree 2-3: Smooth curves (usually good)
   - High degree: Overfits (wiggly, unstable)

5. OPTIMIZERS have different characteristics
   - Batch: Smooth but slow
   - SGD: Fast but noisy
   - Mini-Batch: Balanced (best for practice)

6. CLOSED-FORM and GRADIENT DESCENT give same results
   - Closed-form is exact but expensive
   - Gradient descent is approximate but scalable

Key insight: Visualization isn't just pretty pictures — it's essential for
understanding what's happening during training! Always plot your loss curves!
    """)