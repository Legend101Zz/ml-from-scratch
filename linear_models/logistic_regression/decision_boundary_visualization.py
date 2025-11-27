"""
DECISION BOUNDARY VISUALIZATION: Seeing What the Model Learns
=============================================================

This module creates ASCII-based visualizations of logistic regression decision
boundaries. When I first implemented logistic regression, looking at loss values
and metrics felt abstract. But when I visualized the actual decision boundary
separating classes, everything clicked!

Seeing the geometric interpretation made me understand:
- What the weights actually mean (perpendicular to boundary, pointing toward class 1)
- How confident the model is (distance from boundary)
- Where mistakes happen (usually near the boundary)
- How linear boundaries can't capture complex patterns (motivating polynomial features)

THE VISUALIZATIONS:
------------------
1. Decision Boundary: The line/hyperplane separating classes
2. Probability Contours: Lines of equal probability (0.1, 0.3, 0.5, 0.7, 0.9)
3. Decision Regions: Areas classified as each class
4. Data Points: Show which points are correctly/incorrectly classified

These visualizations work best for 2D data (two features). For higher dimensions,
we can plot 2D slices or use dimensionality reduction.

Author: Mrigesh (learning that a picture is worth a thousand gradients)
"""



from logistic_regression_scratch import LogisticRegression

from foundations.linear_algebra.vectors_and_matrices import Matrix


def plot_decision_boundary_2d(
    model: LogisticRegression,
    X: Matrix,
    y: Matrix,
    width: int = 60,
    height: int = 20,
    title: str = "Decision Boundary"
) -> None:
    """
    Plot 2D decision boundary for logistic regression.
    
    This creates an ASCII visualization showing:
    - The decision boundary (where P(y=1) = 0.5)
    - Data points from each class
    - Which points are correctly/incorrectly classified
    
    THE DECISION BOUNDARY EQUATION:
    ------------------------------
    For 2D data with features [x₁, x₂], the boundary is where:
        w₀ + w₁x₁ + w₂x₂ = 0
    
    Solving for x₂:
        x₂ = -(w₀ + w₁x₁) / w₂
    
    This is a straight line! The slope is -w₁/w₂ and intercept is -w₀/w₂.
    
    INTERPRETING THE VISUALIZATION:
    -------------------------------
    - Points marked 'O': Class 0 (correctly classified)
    - Points marked 'X': Class 1 (correctly classified)
    - Points marked 'o': Class 0 (misclassified as class 1)
    - Points marked 'x': Class 1 (misclassified as class 0)
    - The line separating them: Decision boundary
    
    WHAT TO LOOK FOR:
    ----------------
    - Clean separation: Classes are linearly separable
    - Misclassified points near boundary: Normal, classes overlap there
    - Misclassified points far from boundary: Model has issues or outliers
    - Boundary angle: Shows which features are more important
    
    PARAMETERS:
    ----------
    model : LogisticRegression
        Trained logistic regression model
        
    X : Matrix, shape (n_samples, 3)
        Features with bias column: [1, x₁, x₂]
        
    y : Matrix, shape (n_samples, 1)
        True labels (0 or 1)
        
    width : int, default=60
        Plot width in characters
        
    height : int, default=20
        Plot height in characters
        
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
    
    # Get decision boundary parameters
    params = model.get_decision_boundary_params()
    
    if 'slope' not in params or 'intercept' not in params:
        print("Cannot plot decision boundary (requires exactly 2 features)")
        return
    
    slope = params['slope']
    intercept = params['intercept']
    
    # Extract features (skip bias column)
    x1_values = [X[i, 1] for i in range(X.num_rows)]
    x2_values = [X[i, 2] for i in range(X.num_rows)]
    y_values = [int(y[i, 0]) for i in range(y.num_rows)]
    
    # Get predictions
    predictions = model.predict(X)
    pred_values = [int(predictions[i, 0]) for i in range(X.num_rows)]
    
    # Find data ranges
    x1_min, x1_max = min(x1_values), max(x1_values)
    x2_min, x2_max = min(x2_values), max(x2_values)
    
    # Add padding
    x1_range = x1_max - x1_min
    x2_range = x2_max - x2_min
    x1_min -= x1_range * 0.1
    x1_max += x1_range * 0.1
    x2_min -= x2_range * 0.1
    x2_max += x2_range * 0.1
    
    # Create empty plot
    plot = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Helper to convert data coords to plot coords
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
        
        # Choose marker based on class and correctness
        if true_label == 0:
            marker = 'O' if pred_label == 0 else 'o'  # O = correct, o = wrong
        else:
            marker = 'X' if pred_label == 1 else 'x'  # X = correct, x = wrong
        
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
    
    # Print decision boundary equation
    print(f"\nDecision Boundary: x₂ = {slope:.4f} × x₁ + {intercept:.4f}")
    print(f"(Points above line: class 1, below line: class 0)")
    
    # Print statistics
    n_correct = sum(1 for i in range(len(y_values)) 
                   if y_values[i] == pred_values[i])
    accuracy = n_correct / len(y_values)
    
    print(f"\nAccuracy: {accuracy:.4f} ({n_correct}/{len(y_values)} correct)")
    
    print("\nLegend:")
    print("  O = Class 0 (correct)    o = Class 0 (misclassified)")
    print("  X = Class 1 (correct)    x = Class 1 (misclassified)")
    print("  | = Decision boundary")
    print("=" * 70)


def plot_probability_contours(
    model: LogisticRegression,
    X: Matrix,
    y: Matrix,
    contour_levels: list = [0.1, 0.3, 0.5, 0.7, 0.9],
    width: int = 60,
    height: int = 20,
    title: str = "Probability Contours"
) -> None:
    """
    Plot probability contours showing confidence regions.
    
    This visualization shows lines of equal probability. Points on the same
    contour line have the same predicted probability of being class 1.
    
    INTERPRETING THE CONTOURS:
    -------------------------
    - 0.5 contour: The decision boundary (50-50 split)
    - 0.9 contour: Model is 90% confident it's class 1
    - 0.1 contour: Model is 90% confident it's class 0
    
    Distance from the decision boundary indicates confidence:
    - Far from boundary: Model is very confident
    - Near boundary: Model is uncertain
    
    THE MATHEMATICS:
    ---------------
    A contour for probability p is where:
        σ(w^T x) = p
        ⟺ w^T x = log(p / (1-p))
    
    This is still a line! Each probability level defines a parallel line.
    
    PARAMETERS:
    ----------
    model : LogisticRegression
        Trained model
        
    X : Matrix
        Features (with bias)
        
    y : Matrix
        True labels
        
    contour_levels : list, default=[0.1, 0.3, 0.5, 0.7, 0.9]
        Probability levels to plot
        
    width, height : int
        Plot dimensions
        
    title : str
        Plot title
    """
    
    if X.num_cols != 3:
        raise ValueError("This function only works for 2D data")
    
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)
    
    # Get decision boundary parameters
    params = model.get_decision_boundary_params()
    
    if 'weights' not in params:
        print("Cannot plot contours")
        return
    
    w0 = params['bias']
    w1, w2 = params['weights']
    
    # Extract features
    x1_values = [X[i, 1] for i in range(X.num_rows)]
    x2_values = [X[i, 2] for i in range(X.num_rows)]
    y_values = [int(y[i, 0]) for i in range(y.num_rows)]
    
    # Find ranges
    x1_min, x1_max = min(x1_values), max(x1_values)
    x2_min, x2_max = min(x2_values), max(x2_values)
    
    x1_range = x1_max - x1_min
    x2_range = x2_max - x2_min
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
    
    # Draw contours
    import math
    
    contour_chars = {0.1: '1', 0.3: '3', 0.5: '|', 0.7: '7', 0.9: '9'}
    
    for prob in contour_levels:
        # For probability p, solve: w0 + w1*x1 + w2*x2 = log(p/(1-p))
        log_odds = math.log(prob / (1 - prob))
        
        # x2 = (log_odds - w0 - w1*x1) / w2
        if abs(w2) > 1e-10:
            for px in range(width):
                x1 = x1_min + (x1_max - x1_min) * px / (width - 1)
                x2 = (log_odds - w0 - w1 * x1) / w2
                
                if x2_min <= x2 <= x2_max:
                    _, py = to_plot_coords(x1, x2)
                    if 0 <= py < height:
                        char = contour_chars.get(prob, '*')
                        plot[py][px] = char
    
    # Plot data points
    for i in range(X.num_rows):
        x1, x2 = x1_values[i], x2_values[i]
        label = y_values[i]
        
        px, py = to_plot_coords(x1, x2)
        marker = 'O' if label == 0 else 'X'
        
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
    
    print("\nContour Legend:")
    for prob, char in sorted(contour_chars.items()):
        if prob in contour_levels:
            print(f"  {char} = P(y=1) = {prob:.1f}")
    
    print("\nData Legend:")
    print("  O = Class 0")
    print("  X = Class 1")
    print("=" * 70)


def print_decision_summary(
    model: LogisticRegression,
    X: Matrix,
    y: Matrix
) -> None:
    """
    Print a text summary of the decision boundary and model performance.
    
    This provides a quick textual overview without plotting, useful for
    understanding the model's behavior.
    
    WHAT'S INCLUDED:
    ---------------
    - Weight vector magnitude and direction
    - Decision boundary equation
    - Classification performance
    - Examples of confident and uncertain predictions
    
    PARAMETERS:
    ----------
    model : LogisticRegression
        Trained model
        
    X : Matrix
        Features
        
    y : Matrix
        True labels
    """
    
    print("\n" + "=" * 70)
    print("DECISION BOUNDARY SUMMARY")
    print("=" * 70)
    
    # Get parameters
    params = model.get_decision_boundary_params()
    
    print("\nModel Parameters:")
    print(f"  Bias (w₀): {params['bias']:.4f}")
    
    if 'weights' in params:
        weights = params['weights']
        print(f"  Feature weights: {[f'{w:.4f}' for w in weights]}")
        
        # Weight magnitude (excluding bias)
        magnitude = sum(w**2 for w in weights) ** 0.5
        print(f"  Weight magnitude: {magnitude:.4f}")
    
    if 'slope' in params and 'intercept' in params:
        print(f"\nDecision Boundary (for 2D):")
        print(f"  x₂ = {params['slope']:.4f} × x₁ + {params['intercept']:.4f}")
    
    # Performance
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    correct = sum(1 for i in range(y.num_rows)
                 if abs(predictions[i, 0] - y[i, 0]) < 0.5)
    accuracy = correct / y.num_rows
    
    print(f"\nClassification Performance:")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{y.num_rows})")
    
    # Find most/least confident predictions
    confidences = []
    for i in range(y.num_rows):
        prob = probabilities[i, 0]
        # Confidence is distance from 0.5
        conf = abs(prob - 0.5)
        confidences.append((i, conf, prob, y[i, 0]))
    
    confidences.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nMost Confident Predictions:")
    for i in range(min(3, len(confidences))):
        idx, conf, prob, true_label = confidences[i]
        pred_label = 1 if prob >= 0.5 else 0
        correct_str = "✓" if pred_label == true_label else "✗"
        print(f"  Sample {idx}: P(y=1) = {prob:.4f}, True = {int(true_label)}, "
              f"Pred = {pred_label} {correct_str}")
    
    print(f"\nLeast Confident Predictions (near boundary):")
    for i in range(max(0, len(confidences) - 3), len(confidences)):
        idx, conf, prob, true_label = confidences[i]
        pred_label = 1 if prob >= 0.5 else 0
        correct_str = "✓" if pred_label == true_label else "✗"
        print(f"  Sample {idx}: P(y=1) = {prob:.4f}, True = {int(true_label)}, "
              f"Pred = {pred_label} {correct_str}")
    
    print("=" * 70)