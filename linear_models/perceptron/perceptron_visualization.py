"""
PERCEPTRON VISUALIZATION: Watching the Algorithm Learn
======================================================

This module creates visualizations to see the perceptron learn in real-time.
When I first implemented the perceptron, the algorithm seemed abstract. But
watching the decision boundary rotate and shift as it corrected mistakes made
everything click!

Seeing the learning process visually taught me:
- How each mistake nudges the boundary
- Why it converges for separable data
- Why it fails for non-separable data
- The geometric meaning of the weight vector

These visualizations work best for 2D data (two features). For higher dimensions,
we'd need dimensionality reduction or 2D slices.

Author: Mrigesh (learning by seeing)
"""


from perceptron import Perceptron

from foundations.linear_algebra.vectors_and_matrices import Matrix


def plot_decision_boundary_2d(
    perceptron: Perceptron,
    X: Matrix,
    y: Matrix,
    width: int = 60,
    height: int = 20,
    title: str = "Perceptron Decision Boundary"
) -> None:
    """
    Plot 2D decision boundary for perceptron.
    
    This creates an ASCII visualization showing:
    - The linear decision boundary (where w^T x = 0)
    - Data points from each class
    - Which points are correctly/incorrectly classified
    
    THE DECISION BOUNDARY:
    ---------------------
    For 2D data with features [x₁, x₂], the boundary is:
        w₀ + w₁x₁ + w₂x₂ = 0
        x₂ = -(w₀ + w₁x₁) / w₂
    
    This is a straight line. Points on one side are class 0, other side class 1.
    
    INTERPRETING THE PLOT:
    ---------------------
    - 'O': Class 0 samples (correctly classified)
    - 'X': Class 1 samples (correctly classified)
    - 'o': Class 0 samples (misclassified)
    - 'x': Class 1 samples (misclassified)
    - '|': The decision boundary
    
    WHAT TO LOOK FOR:
    ----------------
    If converged:
    - Clean separation between classes
    - All points correctly classified
    - Boundary cleanly divides the space
    
    If not converged:
    - Misclassified points on wrong side
    - May indicate data isn't linearly separable
    
    PARAMETERS:
    ----------
    perceptron : Perceptron
        Trained perceptron model
        
    X : Matrix, shape (n_samples, 3)
        Features with bias: [1, x₁, x₂]
        
    y : Matrix, shape (n_samples, 1)
        True labels (0 or 1)
        
    width, height : int
        Plot dimensions in characters
        
    title : str
        Plot title
    """
    
    if X.num_cols != 3:
        raise ValueError(
            "This function only works for 2D data (2 features + bias). "
            f"Got {X.num_cols - 1} features."
        )
    
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)
    
    # Get boundary parameters
    params = perceptron.get_decision_boundary_params()
    
    if 'slope' not in params or 'intercept' not in params:
        print("Cannot plot boundary (requires exactly 2 features)")
        return
    
    slope = params['slope']
    intercept = params['intercept']
    
    # Extract features (skip bias)
    x1_values = [X[i, 1] for i in range(X.num_rows)]
    x2_values = [X[i, 2] for i in range(X.num_rows)]
    y_values = [int(y[i, 0]) for i in range(y.num_rows)]
    
    # Get predictions
    predictions = perceptron.predict(X)
    pred_values = [int(predictions[i, 0]) for i in range(X.num_rows)]
    
    # Find data ranges
    x1_min, x1_max = min(x1_values), max(x1_values)
    x2_min, x2_max = min(x2_values), max(x2_values)
    
    # Add padding
    x1_range = x1_max - x1_min if x1_max != x1_min else 1.0
    x2_range = x2_max - x2_min if x2_max != x2_min else 1.0
    x1_min -= x1_range * 0.1
    x1_max += x1_range * 0.1
    x2_min -= x2_range * 0.1
    x2_max += x2_range * 0.1
    
    # Create plot
    plot = [[' ' for _ in range(width)] for _ in range(height)]
    
    def to_plot_coords(x1, x2):
        px = int((x1 - x1_min) / (x1_max - x1_min) * (width - 1))
        py = height - 1 - int((x2 - x2_min) / (x2_max - x2_min) * (height - 1))
        return max(0, min(width - 1, px)), max(0, min(height - 1, py))
    
    # Draw decision boundary
    for px in range(width):
        x1 = x1_min + (x1_max - x1_min) * px / (width - 1)
        x2 = slope * x1 + intercept
        
        if x2_min <= x2 <= x2_max:
            _, py = to_plot_coords(x1, x2)
            if 0 <= py < height:
                plot[py][px] = '|'
    
    # Plot data points
    for i in range(X.num_rows):
        x1, x2 = x1_values[i], x2_values[i]
        true_label = y_values[i]
        pred_label = pred_values[i]
        
        px, py = to_plot_coords(x1, x2)
        
        # Choose marker
        if true_label == 0:
            marker = 'O' if pred_label == 0 else 'o'
        else:
            marker = 'X' if pred_label == 1 else 'x'
        
        if 0 <= px < width and 0 <= py < height:
            plot[py][px] = marker
    
    # Print plot
    print(f"\n{x2_max:7.2f} +", end='')
    for row in plot:
        print(''.join(row))
        if row != plot[-1]:
            print("        |", end='')
    
    print(f"{x2_min:7.2f} +")
    print("        " + "-" * width)
    print(f"        {x1_min:7.2f}" + " " * (width - 16) + f"{x1_max:7.2f}")
    
    # Print info
    print(f"\nDecision Boundary: x₂ = {slope:.4f} × x₁ + {intercept:.4f}")
    
    n_correct = sum(1 for i in range(len(y_values)) 
                   if y_values[i] == pred_values[i])
    accuracy = n_correct / len(y_values)
    
    print(f"Accuracy: {accuracy:.4f} ({n_correct}/{len(y_values)})")
    
    if perceptron.converged_:
        print("\n✓ Model converged (perfect separation)")
    else:
        print(f"\n✗ Model did not converge ({len(y_values) - n_correct} mistakes remain)")
    
    print("\nLegend:")
    print("  O = Class 0 (correct)    o = Class 0 (wrong)")
    print("  X = Class 1 (correct)    x = Class 1 (wrong)")
    print("  | = Decision boundary")
    print("=" * 70)


def plot_learning_curve(
    perceptron: Perceptron,
    title: str = "Perceptron Learning Curve"
) -> None:
    """
    Plot how mistakes decrease during training.
    
    This shows the learning progress:
    - Y-axis: Number of mistakes per epoch
    - X-axis: Epoch number
    
    WHAT TO LOOK FOR:
    ----------------
    Converged:
    - Mistakes decrease to zero
    - Clear downward trend
    - Reaches perfect accuracy
    
    Not converged:
    - Mistakes never reach zero
    - May oscillate (cycle through same mistakes)
    - Indicates non-separable data
    
    PARAMETERS:
    ----------
    perceptron : Perceptron
        Trained perceptron with history
        
    title : str
        Plot title
    """
    
    if perceptron.history_ is None:
        print("No training history available")
        return
    
    mistakes = perceptron.history_['mistakes']
    epochs = perceptron.history_['epoch']
    
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)
    
    if not mistakes:
        print("No training history to plot")
        return
    
    # Find max mistakes for scaling
    max_mistakes = max(mistakes)
    
    # Create ASCII plot
    height = 15
    width = 60
    
    plot = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Plot the learning curve
    for i, (epoch, mistake_count) in enumerate(zip(epochs, mistakes)):
        if len(epochs) == 1:
            px = 0
        else:
            px = int(i / (len(epochs) - 1) * (width - 1))
        
        if max_mistakes > 0:
            py = height - 1 - int(mistake_count / max_mistakes * (height - 1))
        else:
            py = height - 1
        
        if 0 <= px < width and 0 <= py < height:
            plot[py][px] = '*'
    
    # Print plot
    print(f"\n{max_mistakes:7.0f} |", end='')
    for row in plot:
        print(''.join(row))
        if row != plot[-1]:
            print("        |", end='')
    
    print(f"{0:7.0f} |")
    print("        " + "-" * width)
    print(f"        Epoch 0" + " " * (width - 20) + f"Epoch {max(epochs)}")
    
    # Print statistics
    print(f"\nInitial mistakes: {mistakes[0]}")
    print(f"Final mistakes: {mistakes[-1]}")
    
    if perceptron.converged_:
        converged_epoch = len([m for m in mistakes if m > 0])
        print(f"Converged at epoch: {converged_epoch}")
    else:
        print("Did not converge")
    
    print(f"Total updates: {sum(mistakes)}")
    print("=" * 70)


def print_weight_evolution(
    perceptron: Perceptron
) -> None:
    """
    Print summary of final learned weights.
    
    This helps understand what the perceptron learned:
    - Which features are important (large weights)
    - Direction of decision boundary (weight signs)
    - Bias term (shifts boundary)
    
    INTERPRETING WEIGHTS:
    --------------------
    For binary classification:
    - Positive weight: Feature increases probability of class 1
    - Negative weight: Feature increases probability of class 0
    - Large magnitude: Feature is important
    - Small magnitude: Feature is less important
    
    The weight vector w is perpendicular to the decision boundary and
    points toward the class 1 region.
    
    PARAMETERS:
    ----------
    perceptron : Perceptron
        Trained perceptron
    """
    
    if perceptron.weights_ is None:
        print("Model not trained")
        return
    
    print("\n" + "=" * 70)
    print("LEARNED WEIGHTS")
    print("=" * 70)
    
    params = perceptron.get_decision_boundary_params()
    
    print(f"\nBias (w₀): {params['bias']:.4f}")
    
    if 'weights' in params:
        weights = params['weights']
        print("\nFeature weights:")
        for i, w in enumerate(weights, 1):
            direction = "→ class 1" if w > 0 else "→ class 0"
            magnitude = "strong" if abs(w) > 1 else "weak"
            print(f"  w{i} = {w:7.4f}  ({magnitude} influence, {direction})")
        
        # Weight magnitude
        magnitude = sum(w**2 for w in weights) ** 0.5
        print(f"\nWeight magnitude: {magnitude:.4f}")
        print("(Larger magnitude = steeper decision boundary)")
    
    if 'slope' in params and 'intercept' in params:
        print(f"\nDecision boundary (2D):")
        print(f"  x₂ = {params['slope']:.4f} × x₁ + {params['intercept']:.4f}")
    
    print("=" * 70)


def compare_with_logistic_regression(
    X: Matrix,
    y: Matrix,
    title: str = "Perceptron vs Logistic Regression"
) -> None:
    """
    Train both perceptron and logistic regression, compare results.
    
    This visualization shows the key differences:
    - Perceptron: Hard decision boundary, no probabilities
    - Logistic: Soft boundary with probability gradients
    
    WHAT YOU'LL SEE:
    ---------------
    - Both find linear boundaries
    - Perceptron: Any separator (if separable)
    - Logistic: Best fit separator (maximizes margin-like property)
    - Perceptron: Fast, simple
    - Logistic: Slower but more principled
    
    PARAMETERS:
    ----------
    X : Matrix
        Features (with bias)
        
    y : Matrix
        Labels
        
    title : str
        Comparison title
    """
    
    from logistic_regression.logistic_regression_scratch import \
        LogisticRegression
    
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)
    
    # Train perceptron
    print("\nTraining Perceptron...")
    perc = Perceptron(learning_rate=1.0, n_epochs=100, verbose=False)
    perc.fit(X, y)
    perc_acc = perc.score(X, y)
    
    # Train logistic regression
    print("Training Logistic Regression...")
    log_reg = LogisticRegression(learning_rate=0.1, n_epochs=200, verbose=False)
    log_reg.fit(X, y)
    log_reg_acc = log_reg.score(X, y)
    
    # Compare
    print("\n" + "-" * 70)
    print("COMPARISON")
    print("-" * 70)
    
    print(f"\nPerceptron:")
    print(f"  Accuracy: {perc_acc:.4f}")
    print(f"  Converged: {'Yes' if perc.converged_ else 'No'}")
    print(f"  Updates: {sum(perc.history_['mistakes'])}")
    
    print(f"\nLogistic Regression:")
    print(f"  Accuracy: {log_reg_acc:.4f}")
    print(f"  Final loss: {log_reg.history_['loss'][-1]:.6f}")
    
    # Decision boundaries
    perc_params = perc.get_decision_boundary_params()
    log_params = log_reg.get_decision_boundary_params()
    
    if 'slope' in perc_params and 'slope' in log_params:
        print(f"\nDecision Boundaries (2D):")
        print(f"  Perceptron: x₂ = {perc_params['slope']:.4f}x₁ + {perc_params['intercept']:.4f}")
        print(f"  Logistic:   x₂ = {log_params['slope']:.4f}x₁ + {log_params['intercept']:.4f}")
        
        if abs(perc_params['slope'] - log_params['slope']) < 0.1:
            print("\n  → Very similar boundaries!")
        else:
            print("\n  → Different boundaries (both can be correct)")
    
    print("\nKey Differences:")
    print("  Perceptron: Step activation, hard decisions, updates only on mistakes")
    print("  Logistic: Sigmoid activation, probabilities, gradient-based updates")
    print("=" * 70)