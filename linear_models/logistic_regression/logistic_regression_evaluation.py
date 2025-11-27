"""
LOGISTIC REGRESSION EVALUATION: Beyond Accuracy
================================================

This module implements evaluation metrics and tools specifically for binary
classification. When I first started with logistic regression, I only looked at
accuracy. Then I built a model with 95% accuracy that was completely useless
because it just predicted the majority class for everything!

That painful lesson taught me that classification evaluation requires much more
nuance than regression. This module contains all the tools I wish I had from
the beginning.

THE CLASSIFICATION EVALUATION TOOLKIT:
-------------------------------------
1. Confusion Matrix: The foundation - shows all four outcomes (TP, TN, FP, FN)
2. Precision, Recall, F1: Handle imbalanced data properly
3. ROC Curve and AUC: Threshold-independent performance measure
4. Precision-Recall Curve: Better than ROC for imbalanced data
5. Classification Report: Comprehensive summary of all metrics

THE CONFUSION MATRIX:
--------------------
This is the KEY to understanding classification performance:

                    Predicted
                    0      1
    Actual    0    TN     FP    (False Positive = Type I Error)
              1    FN     TP    (True Positive)

From these four numbers, we can compute every classification metric:
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP) - "Of predicted positives, how many were correct?"
- Recall = TP / (TP + FN) - "Of actual positives, how many did we find?"
- F1 Score = 2 × (Precision × Recall) / (Precision + Recall)

Understanding this matrix deeply was transformative for my ML journey.

Author: Mrigesh (learning that proper evaluation is as important as the algorithm)
"""



from typing import List, Tuple

from foundations.linear_algebra.vectors_and_matrices import Matrix
from foundations.loss_functions.metrics import (ConfusionMatrix, accuracy,
                                                f1_score, precision, recall)


def compute_roc_curve(
    y_true: Matrix,
    y_proba: Matrix,
    n_thresholds: int = 100
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute ROC (Receiver Operating Characteristic) curve.
    
    The ROC curve plots True Positive Rate vs False Positive Rate as we vary
    the decision threshold. It shows the tradeoff between catching positives
    (sensitivity) and avoiding false alarms (specificity).
    
    THE INTUITION:
    -------------
    Imagine a medical test for disease. We can adjust how "strict" the test is:
    
    Very strict (high threshold, e.g., 0.9):
    - Only diagnose disease when very confident
    - Few false positives (low FPR) - don't wrongly diagnose healthy people
    - But also few true positives (low TPR) - miss some actual cases
    
    Very lenient (low threshold, e.g., 0.1):
    - Diagnose disease even when somewhat uncertain
    - Many true positives (high TPR) - catch most actual cases
    - But also many false positives (high FPR) - wrongly diagnose healthy people
    
    The ROC curve shows this tradeoff across all possible thresholds!
    
    THE PERFECT CLASSIFIER:
    ----------------------
    Would have TPR = 1.0 (catches all positives) and FPR = 0.0 (no false alarms).
    The ROC curve would go straight up the left side then across the top.
    
    THE RANDOM CLASSIFIER:
    ---------------------
    Would have TPR = FPR (equal hit rate and false alarm rate).
    The ROC curve would be the diagonal line from (0,0) to (1,1).
    
    OUR CLASSIFIER:
    --------------
    Should be somewhere in between - the more the curve "bows" toward the top-left
    corner, the better the classifier.
    
    THE ALGORITHM:
    -------------
    For threshold in [0, 1]:
        predictions = (probabilities >= threshold)
        TPR = TP / (TP + FN)  # True Positive Rate (Recall)
        FPR = FP / (FP + TN)  # False Positive Rate
        Record (FPR, TPR) point
    
    Plot these points to get the ROC curve.
    
    PARAMETERS:
    ----------
    y_true : Matrix, shape (n_samples, 1)
        True binary labels (0 or 1)
        
    y_proba : Matrix, shape (n_samples, 1)
        Predicted probabilities for class 1
        
    n_thresholds : int, default=100
        Number of thresholds to try
        
    RETURNS:
    -------
    fpr : List[float]
        False positive rates (x-axis of ROC curve)
        
    tpr : List[float]
        True positive rates (y-axis of ROC curve)
        
    thresholds : List[float]
        Thresholds used to generate each point
    """
    
    if y_true.shape != y_proba.shape:
        raise ValueError("y_true and y_proba must have same shape")
    
    n_samples = y_true.num_rows
    
    # Generate thresholds from 0 to 1
    thresholds = [i / (n_thresholds - 1) for i in range(n_thresholds)]
    
    fpr_list = []
    tpr_list = []
    
    for threshold in thresholds:
        # Make predictions with this threshold
        predictions = []
        for i in range(n_samples):
            pred = 1.0 if y_proba[i, 0] >= threshold else 0.0
            predictions.append([pred])
        
        y_pred = Matrix(predictions)
        
        # Compute confusion matrix
        cm = ConfusionMatrix(y_true, y_pred)
        tp, tn, fp, fn = cm.get_tp_tn_fp_fn(positive_class=1)
        
        # Compute TPR and FPR
        if tp + fn > 0:
            tpr = tp / (tp + fn)
        else:
            tpr = 0.0
        
        if fp + tn > 0:
            fpr = fp / (fp + tn)
        else:
            fpr = 0.0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return fpr_list, tpr_list, thresholds


def compute_auc(fpr: List[float], tpr: List[float]) -> float:
    """
    Compute AUC (Area Under the ROC Curve).
    
    AUC is a single number summarizing the ROC curve. It represents the probability
    that the model ranks a random positive example higher than a random negative
    example.
    
    INTERPRETATION:
    --------------
    - AUC = 1.0: Perfect classifier
    - AUC = 0.9: Excellent
    - AUC = 0.8: Good
    - AUC = 0.7: Fair
    - AUC = 0.6: Poor
    - AUC = 0.5: Random guessing (no better than coin flip)
    - AUC < 0.5: Worse than random (predictions are backwards!)
    
    WHY AUC IS USEFUL:
    -----------------
    Unlike accuracy, precision, or recall, AUC is:
    1. Threshold-independent (measures overall ranking quality)
    2. Handles imbalanced data well
    3. Single number for easy comparison
    
    However, AUC has limitations:
    - Doesn't tell you optimal threshold
    - Can be misleading if you care more about precision or recall
    - Less interpretable than confusion matrix
    
    THE COMPUTATION:
    ---------------
    We use the trapezoidal rule to compute the area under the curve:
    
    AUC = Σᵢ (fprᵢ₊₁ - fprᵢ) × (tprᵢ + tprᵢ₊₁) / 2
    
    This is just summing up trapezoid areas under the curve.
    
    PARAMETERS:
    ----------
    fpr : List[float]
        False positive rates (x-coordinates)
        
    tpr : List[float]
        True positive rates (y-coordinates)
        
    RETURNS:
    -------
    auc : float
        Area under the ROC curve (between 0 and 1)
    """
    
    if len(fpr) != len(tpr):
        raise ValueError("fpr and tpr must have same length")
    
    if len(fpr) < 2:
        raise ValueError("Need at least 2 points to compute area")
    
    # Sort by fpr (x-axis) to ensure proper ordering
    points = sorted(zip(fpr, tpr))
    fpr_sorted = [p[0] for p in points]
    tpr_sorted = [p[1] for p in points]
    
    # Trapezoidal rule: sum of trapezoid areas
    auc = 0.0
    for i in range(len(fpr_sorted) - 1):
        # Width of trapezoid
        width = fpr_sorted[i + 1] - fpr_sorted[i]
        
        # Average height of trapezoid
        avg_height = (tpr_sorted[i] + tpr_sorted[i + 1]) / 2
        
        # Area of this trapezoid
        auc += width * avg_height
    
    return auc


def compute_precision_recall_curve(
    y_true: Matrix,
    y_proba: Matrix,
    n_thresholds: int = 100
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute Precision-Recall curve.
    
    This plots Precision vs Recall as we vary the decision threshold. It's often
    more informative than ROC curve for imbalanced datasets.
    
    WHY PRECISION-RECALL INSTEAD OF ROC?
    -----------------------------------
    For highly imbalanced data (e.g., 1% positive class), ROC curves can be
    misleadingly optimistic because they include the True Negative Rate.
    
    With 99% negatives, even a mediocre classifier will have high True Negative
    Rate, making the ROC curve look good. But precision (which focuses on the
    positive predictions) will reveal poor performance.
    
    Precision-Recall curves focus on the minority class performance and are more
    informative when that's what you care about.
    
    THE TRADEOFF:
    ------------
    High threshold (e.g., 0.9):
    - High precision (most predicted positives are correct)
    - Low recall (miss many actual positives)
    
    Low threshold (e.g., 0.1):
    - Low precision (many false positives)
    - High recall (catch most actual positives)
    
    The curve shows this tradeoff. The area under this curve is another measure
    of overall performance.
    
    PARAMETERS:
    ----------
    y_true : Matrix
        True labels
        
    y_proba : Matrix
        Predicted probabilities
        
    n_thresholds : int, default=100
        Number of thresholds to try
        
    RETURNS:
    -------
    precision_list : List[float]
        Precision at each threshold
        
    recall_list : List[float]
        Recall at each threshold
        
    thresholds : List[float]
        Thresholds used
    """
    
    if y_true.shape != y_proba.shape:
        raise ValueError("y_true and y_proba must have same shape")
    
    n_samples = y_true.num_rows
    thresholds = [i / (n_thresholds - 1) for i in range(n_thresholds)]
    
    precision_list = []
    recall_list = []
    
    for threshold in thresholds:
        # Make predictions
        predictions = []
        for i in range(n_samples):
            pred = 1.0 if y_proba[i, 0] >= threshold else 0.0
            predictions.append([pred])
        
        y_pred = Matrix(predictions)
        
        # Compute precision and recall
        prec = precision(y_true, y_pred, positive_class=1)
        rec = recall(y_true, y_pred, positive_class=1)
        
        precision_list.append(prec)
        recall_list.append(rec)
    
    return precision_list, recall_list, thresholds


def find_optimal_threshold(
    y_true: Matrix,
    y_proba: Matrix,
    metric: str = 'f1',
    n_thresholds: int = 100
) -> Tuple[float, float]:
    """
    Find the optimal decision threshold based on a chosen metric.
    
    The default threshold of 0.5 isn't always optimal! This function searches
    for the threshold that maximizes a given metric.
    
    SUPPORTED METRICS:
    -----------------
    - 'f1': Maximize F1 score (balance precision and recall)
    - 'precision': Maximize precision (minimize false positives)
    - 'recall': Maximize recall (minimize false negatives)
    - 'accuracy': Maximize accuracy (might not be meaningful for imbalanced data)
    
    WHEN TO USE EACH:
    ----------------
    F1: When you care about precision and recall equally
    Precision: When false positives are very costly
    Recall: When false negatives are very costly
    Accuracy: Only when classes are balanced
    
    PARAMETERS:
    ----------
    y_true : Matrix
        True labels
        
    y_proba : Matrix
        Predicted probabilities
        
    metric : str, default='f1'
        Metric to optimize ('f1', 'precision', 'recall', or 'accuracy')
        
    n_thresholds : int, default=100
        Number of thresholds to try
        
    RETURNS:
    -------
    best_threshold : float
        Threshold that maximizes the metric
        
    best_score : float
        Maximum metric value achieved
    """
    
    if y_true.shape != y_proba.shape:
        raise ValueError("y_true and y_proba must have same shape")
    
    valid_metrics = {'f1', 'precision', 'recall', 'accuracy'}
    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics}, got {metric}")
    
    n_samples = y_true.num_rows
    thresholds = [i / (n_thresholds - 1) for i in range(n_thresholds)]
    
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in thresholds:
        # Make predictions with this threshold
        predictions = []
        for i in range(n_samples):
            pred = 1.0 if y_proba[i, 0] >= threshold else 0.0
            predictions.append([pred])
        
        y_pred = Matrix(predictions)
        
        # Compute the chosen metric
        if metric == 'f1':
            score = f1_score(y_true, y_pred, positive_class=1)
        elif metric == 'precision':
            score = precision(y_true, y_pred, positive_class=1)
        elif metric == 'recall':
            score = recall(y_true, y_pred, positive_class=1)
        elif metric == 'accuracy':
            score = accuracy(y_true, y_pred)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def classification_report_binary(
    y_true: Matrix,
    y_pred: Matrix,
    y_proba: Matrix = None
) -> str:
    """
    Generate a comprehensive classification report for binary classification.
    
    This creates a formatted report showing all the important metrics in one place.
    It's like getting a report card for your classifier!
    
    WHAT'S INCLUDED:
    ---------------
    - Confusion Matrix (TP, TN, FP, FN)
    - Accuracy
    - Precision, Recall, F1 Score for each class
    - Support (number of samples per class)
    - If probabilities provided: AUC score
    
    This gives you the complete picture of model performance at a glance.
    
    PARAMETERS:
    ----------
    y_true : Matrix
        True labels
        
    y_pred : Matrix
        Predicted labels
        
    y_proba : Matrix, optional
        Predicted probabilities (for computing AUC)
        
    RETURNS:
    -------
    report : str
        Formatted classification report
    """
    
    # Compute confusion matrix
    cm = ConfusionMatrix(y_true, y_pred)
    tp, tn, fp, fn = cm.get_tp_tn_fp_fn(positive_class=1)
    
    # Compute metrics
    acc = accuracy(y_true, y_pred)
    prec_0 = precision(y_true, y_pred, positive_class=0)
    rec_0 = recall(y_true, y_pred, positive_class=0)
    f1_0 = f1_score(y_true, y_pred, positive_class=0)
    
    prec_1 = precision(y_true, y_pred, positive_class=1)
    rec_1 = recall(y_true, y_pred, positive_class=1)
    f1_1 = f1_score(y_true, y_pred, positive_class=1)
    
    # Count support
    n_class_0 = sum(1 for i in range(y_true.num_rows) if y_true[i, 0] == 0)
    n_class_1 = sum(1 for i in range(y_true.num_rows) if y_true[i, 0] == 1)
    total = n_class_0 + n_class_1
    
    # Build report
    lines = []
    lines.append("=" * 70)
    lines.append("BINARY CLASSIFICATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("CONFUSION MATRIX:")
    lines.append(str(cm))
    lines.append("")
    lines.append("PER-CLASS METRICS:")
    lines.append(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    lines.append("-" * 70)
    lines.append(f"{'0':<10} {prec_0:>10.4f} {rec_0:>10.4f} {f1_0:>10.4f} {n_class_0:>10d}")
    lines.append(f"{'1':<10} {prec_1:>10.4f} {rec_1:>10.4f} {f1_1:>10.4f} {n_class_1:>10d}")
    lines.append("-" * 70)
    lines.append(f"{'Accuracy':<10} {'':<10} {'':<10} {acc:>10.4f} {total:>10d}")
    
    # Add AUC if probabilities provided
    if y_proba is not None:
        fpr, tpr, _ = compute_roc_curve(y_true, y_proba)
        auc = compute_auc(fpr, tpr)
        lines.append("")
        lines.append(f"AUC-ROC: {auc:.4f}")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)