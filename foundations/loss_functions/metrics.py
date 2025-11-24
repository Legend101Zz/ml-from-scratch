"""
METRICS: Measuring How Well Your Model Actually Works
=====================================================

This module implements evaluation metrics — the ways we measure if our machine
learning models are actually any good! When I first started, I thought "accuracy"
was enough. Then I built a spam classifier that was "95% accurate" but missed
ALL the spam emails. That's when I learned: choosing the right metric is crucial!

MY LEARNING JOURNEY:
-------------------
Story time: I built a fraud detection model with 99% accuracy. Amazing, right?
WRONG! Fraud only happened 1% of the time, so a model that ALWAYS predicted
"not fraud" would also be 99% accurate — but completely useless!

This taught me that accuracy alone can be misleading. I needed to understand:
- Precision: Of the frauds I detected, how many were real?
- Recall: Of the real frauds, how many did I catch?
- F1 score: How do I balance precision and recall?

Different problems need different metrics. There's no one "best" metric — it
depends on what you care about!

THE BIG INSIGHT:
---------------
Metrics aren't just numbers you report at the end. They fundamentally shape what
your model optimizes for! If you measure the wrong thing, you'll build the wrong
model — even if the algorithm is perfect.

WHAT WE'LL IMPLEMENT:
--------------------
1. Accuracy: Overall correctness (simple but often misleading)
2. Precision: When you predict positive, how often are you right?
3. Recall: Of all actual positives, how many did you find?
4. F1 Score: Harmonic mean of precision and recall
5. Confusion Matrix: The full picture of what's happening
6. R² Score: For regression (how well you explain variance)
7. Classification Report: All metrics in one convenient summary

THE CONFUSION MATRIX:
--------------------
This is the KEY to understanding classification! It shows:

                    Predicted
                    No    Yes
    Actual  No     TN    FP    (False Positive = Type I error)
            Yes    FN    TP    (False Negative = Type II error)

- True Positive (TP): Correctly predicted positive
- True Negative (TN): Correctly predicted negative  
- False Positive (FP): Predicted positive, actually negative (false alarm)
- False Negative (FN): Predicted negative, actually positive (missed detection)

All other metrics derive from these four numbers!

Author: Mrigesh (learning that measuring success is as important as achieving it)
"""

import sys
import os


from foundations.linear_algebra.vectors_and_matrices import Matrix
from typing import Dict, List, Optional, Tuple



# ==============================================================================
# CONFUSION MATRIX
# ==============================================================================

class ConfusionMatrix:
    """
    Confusion matrix for binary or multi-class classification.
    
    This is the foundation of all classification metrics! It shows exactly where
    your model is getting things right and wrong.
    
    FOR BINARY CLASSIFICATION:
    -------------------------
                        Predicted
                     Negative  Positive
        Actual  Neg     TN        FP
                Pos     FN        TP
    
    - True Negatives (TN): Correctly predicted negative
    - False Positives (FP): Incorrectly predicted positive (Type I error)
    - False Negatives (FN): Incorrectly predicted negative (Type II error)  
    - True Positives (TP): Correctly predicted positive
    
    THE ERRORS HAVE NAMES:
    ---------------------
    - Type I Error (FP): False alarm — predicted disease when healthy
    - Type II Error (FN): Missed detection — predicted healthy when diseased
    
    Which is worse depends on context:
    - Cancer screening: Type II errors are catastrophic (miss a cancer case!)
    - Spam filter: Type I errors are annoying (false alarms block real emails)
    
    FOR MULTI-CLASS:
    ---------------
    The matrix is K×K where K is the number of classes.
    Element [i,j] = number of samples with true class i, predicted as class j.
    
    Diagonal elements are correct predictions, off-diagonal are errors.
    
    EXAMPLE USAGE:
    -------------
    >>> y_true = Matrix([[1], [0], [1], [1], [0]])
    >>> y_pred = Matrix([[1], [0], [1], [0], [0]])  # One false negative
    >>> 
    >>> cm = ConfusionMatrix(y_true, y_pred)
    >>> print(cm.matrix.elements)
    [[2, 0],   # Row 0 (actual negative): 2 TN, 0 FP
     [1, 2]]   # Row 1 (actual positive): 1 FN, 2 TP
    """
    
    def __init__(self, y_true: Matrix, y_pred: Matrix):
        """
        Compute confusion matrix from true and predicted labels.
        
        PARAMETERS:
        ----------
        y_true : Matrix, shape (n_samples, 1)
            True class labels (integers 0, 1, 2, ...)
            
        y_pred : Matrix, shape (n_samples, 1)
            Predicted class labels (integers 0, 1, 2, ...)
        """
        
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true is {y_true.shape}, y_pred is {y_pred.shape}"
            )
        
        if y_true.num_cols != 1:
            raise ValueError("y_true and y_pred must be column vectors")
        
        # Determine number of classes
        true_labels = [int(y_true[i, 0]) for i in range(y_true.num_rows)]
        pred_labels = [int(y_pred[i, 0]) for i in range(y_pred.num_rows)]
        
        self.n_classes = max(max(true_labels), max(pred_labels)) + 1
        self.n_samples = y_true.num_rows
        
        # Initialize confusion matrix (all zeros)
        matrix_data = [[0 for _ in range(self.n_classes)] for _ in range(self.n_classes)]
        
        # Fill in the confusion matrix
        for i in range(self.n_samples):
            true_class = int(y_true[i, 0])
            pred_class = int(y_pred[i, 0])
            matrix_data[true_class][pred_class] += 1
        
        self.matrix = Matrix([[float(x) for x in row] for row in matrix_data])
        self.y_true = y_true
        self.y_pred = y_pred
    
    def get_tp_tn_fp_fn(self, positive_class: int = 1) -> Tuple[int, int, int, int]:
        """
        Get TP, TN, FP, FN for binary classification or one-vs-rest.
        
        For binary classification (2 classes), this is straightforward.
        For multi-class, we treat specified class as "positive" and rest as "negative".
        
        THE CALCULATION:
        ---------------
        TP = matrix[positive_class, positive_class]
        FN = sum of row positive_class, excluding TP
        FP = sum of column positive_class, excluding TP
        TN = everything else
        
        PARAMETERS:
        ----------
        positive_class : int
            Which class to treat as "positive" (default: 1)
            
        RETURNS:
        -------
        tp : int
            True positives
        tn : int
            True negatives
        fp : int
            False positives  
        fn : int
            False negatives
        """
        
        if positive_class >= self.n_classes:
            raise ValueError(
                f"positive_class {positive_class} >= n_classes {self.n_classes}"
            )
        
        # True Positives: correctly predicted as positive_class
        tp = int(self.matrix[positive_class, positive_class])
        
        # False Negatives: actually positive_class, predicted as something else
        fn = 0
        for j in range(self.n_classes):
            if j != positive_class:
                fn += int(self.matrix[positive_class, j])
        
        # False Positives: actually something else, predicted as positive_class
        fp = 0
        for i in range(self.n_classes):
            if i != positive_class:
                fp += int(self.matrix[i, positive_class])
        
        # True Negatives: everything else
        tn = self.n_samples - tp - fn - fp
        
        return tp, tn, fp, fn
    
    def __str__(self) -> str:
        """
        Pretty print the confusion matrix.
        """
        lines = []
        lines.append("Confusion Matrix:")
        lines.append("=" * 50)
        
        # Header
        header = "Actual\\Pred |"
        for j in range(self.n_classes):
            header += f" {j:4d} |"
        lines.append(header)
        lines.append("-" * len(header))
        
        # Rows
        for i in range(self.n_classes):
            row_str = f"     {i:4d}   |"
            for j in range(self.n_classes):
                row_str += f" {int(self.matrix[i, j]):4d} |"
            lines.append(row_str)
        
        # Add interpretation for binary case
        if self.n_classes == 2:
            tp, tn, fp, fn = self.get_tp_tn_fp_fn(positive_class=1)
            lines.append("")
            lines.append(f"True Negatives (TN): {tn}")
            lines.append(f"False Positives (FP): {fp} (Type I error)")
            lines.append(f"False Negatives (FN): {fn} (Type II error)")
            lines.append(f"True Positives (TP): {tp}")
        
        return "\n".join(lines)


# ==============================================================================
# BASIC CLASSIFICATION METRICS
# ==============================================================================

def accuracy(y_true: Matrix, y_pred: Matrix) -> float:
    """
    Accuracy: Fraction of correct predictions.
    
    This is the simplest metric: How many predictions did you get right?
    
    THE FORMULA:
    -----------
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
             = Number of correct predictions / Total predictions
    
    WHEN IT'S MISLEADING:
    --------------------
    Accuracy seems intuitive but can be very misleading with imbalanced data!
    
    Example: Fraud detection where 1% of transactions are fraud:
    - Model that always predicts "not fraud": 99% accurate!
    - Model that catches 50% of fraud: maybe only 98% accurate
    - But the second model is much more useful!
    
    THE ACCURACY PARADOX:
    --------------------
    Sometimes a less accurate model is more valuable than a more accurate one,
    if the less accurate model makes better predictions on the class you care about.
    
    WHEN TO USE ACCURACY:
    --------------------
    - Balanced datasets (roughly equal class sizes)
    - When all errors are equally bad
    - When you care about overall correctness
    
    WHEN NOT TO USE:
    ---------------
    - Imbalanced datasets (use precision/recall/F1 instead)
    - When some errors are worse than others
    - When you care more about one class than another
    
    PARAMETERS:
    ----------
    y_true : Matrix, shape (n_samples, 1)
        True labels
        
    y_pred : Matrix, shape (n_samples, 1)
        Predicted labels
        
    RETURNS:
    -------
    accuracy : float
        Fraction of correct predictions (between 0 and 1)
        
    EXAMPLE:
    -------
    >>> y_true = Matrix([[1], [0], [1], [1], [0]])
    >>> y_pred = Matrix([[1], [0], [1], [0], [0]])  # 4/5 correct
    >>> print(accuracy(y_true, y_pred))
    0.8  # 80% accurate
    """
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")
    
    n_samples = y_true.num_rows
    n_correct = 0
    
    for i in range(n_samples):
        if abs(y_true[i, 0] - y_pred[i, 0]) < 1e-9:  # Account for floating point
            n_correct += 1
    
    return n_correct / n_samples


def precision(y_true: Matrix, y_pred: Matrix, positive_class: int = 1) -> float:
    """
    Precision: Of all positive predictions, how many were actually positive?
    
    Also called "Positive Predictive Value" (PPV). This answers: "When my model
    says something is positive, how often is it right?"
    
    THE FORMULA:
    -----------
    Precision = TP / (TP + FP)
              = True Positives / All Positive Predictions
    
    THE INTUITION:
    -------------
    Precision is about being CAREFUL. High precision means when you predict positive,
    you're usually right — few false alarms.
    
    Example: Medical test with high precision
    - When test says "disease," patient usually has disease
    - But might miss many cases (low recall)
    
    PRECISION VS RECALL TRADEOFF:
    ----------------------------
    There's always a tradeoff! You can make precision perfect by only predicting
    positive when you're 100% certain. But then you'll miss lots of positives
    (low recall).
    
    Example: Email spam filter
    - High precision: Few legitimate emails marked as spam (good!)
    - But might miss some actual spam (lower recall)
    
    WHEN HIGH PRECISION MATTERS:
    ---------------------------
    - False positives are costly
    - You want to be sure when you predict positive
    - Example: Recommending surgery — better be sure there's a problem!
    
    EDGE CASE:
    ---------
    If TP + FP = 0 (no positive predictions at all), precision is undefined.
    We return 0.0 by convention.
    
    PARAMETERS:
    ----------
    y_true : Matrix
        True labels
        
    y_pred : Matrix
        Predicted labels
        
    positive_class : int
        Which class to treat as "positive" (default: 1)
        
    RETURNS:
    -------
    precision : float
        Between 0 and 1 (higher is better)
        
    EXAMPLE:
    -------
    >>> # 5 samples, predict 3 as positive, 2 are actually positive
    >>> y_true = Matrix([[1], [0], [1], [0], [1]])
    >>> y_pred = Matrix([[1], [1], [1], [0], [0]])  # Predicted 3 positive
    >>> 
    >>> print(precision(y_true, y_pred))
    0.667  # 2 out of 3 predictions were correct (TP=2, FP=1)
    """
    
    cm = ConfusionMatrix(y_true, y_pred)
    tp, tn, fp, fn = cm.get_tp_tn_fp_fn(positive_class)
    
    if tp + fp == 0:
        # No positive predictions at all
        return 0.0
    
    return tp / (tp + fp)


def recall(y_true: Matrix, y_pred: Matrix, positive_class: int = 1) -> float:
    """
    Recall: Of all actual positives, how many did we find?
    
    Also called "Sensitivity" or "True Positive Rate" (TPR). This answers: "Of
    all the positive cases, how many did my model catch?"
    
    THE FORMULA:
    -----------
    Recall = TP / (TP + FN)
           = True Positives / All Actual Positives
    
    THE INTUITION:
    -------------
    Recall is about being THOROUGH. High recall means you catch most of the
    positive cases — you don't miss many.
    
    Example: Medical screening with high recall
    - Catches most/all disease cases
    - But might have many false alarms (low precision)
    
    THE CANCER SCREENING ANALOGY:
    ----------------------------
    Imagine a cancer screening test:
    
    - High recall (95%): Catches 95 out of 100 cancer cases → Good!
    - But maybe 20% of healthy people test positive (low precision) → Lots of false alarms
    
    For cancer, high recall is more important — missing a cancer case is catastrophic!
    False alarms can be resolved with follow-up tests.
    
    WHEN HIGH RECALL MATTERS:
    ------------------------
    - False negatives are costly
    - You want to catch all/most positive cases
    - Example: Detecting fraud, disease, security threats
    
    THE TRADEOFF:
    ------------
    You can get perfect recall by predicting everything as positive! But then
    precision would be terrible. The art is balancing both.
    
    EDGE CASE:
    ---------
    If TP + FN = 0 (no actual positives), recall is undefined. We return 0.0.
    
    PARAMETERS:
    ----------
    y_true : Matrix
        True labels
        
    y_pred : Matrix
        Predicted labels
        
    positive_class : int
        Which class to treat as "positive" (default: 1)
        
    RETURNS:
    -------
    recall : float
        Between 0 and 1 (higher is better)
        
    EXAMPLE:
    -------
    >>> # 5 samples, 3 are actually positive, we predicted 2 of them
    >>> y_true = Matrix([[1], [0], [1], [0], [1]])
    >>> y_pred = Matrix([[1], [0], [1], [0], [0]])  # Missed the last one!
    >>> 
    >>> print(recall(y_true, y_pred))
    0.667  # Found 2 out of 3 actual positives (TP=2, FN=1)
    """
    
    cm = ConfusionMatrix(y_true, y_pred)
    tp, tn, fp, fn = cm.get_tp_tn_fp_fn(positive_class)
    
    if tp + fn == 0:
        # No actual positives
        return 0.0
    
    return tp / (tp + fn)


def f1_score(y_true: Matrix, y_pred: Matrix, positive_class: int = 1) -> float:
    """
    F1 Score: Harmonic mean of precision and recall.
    
    When you care about both precision AND recall but need a single number,
    F1 score is your answer! It balances both metrics.
    
    THE FORMULA:
    -----------
    F1 = 2 × (Precision × Recall) / (Precision + Recall)
       = 2TP / (2TP + FP + FN)
    
    WHY HARMONIC MEAN?
    -----------------
    We use harmonic mean (not arithmetic mean) because it's more conservative.
    The harmonic mean is dominated by the smaller value:
    
    Example: Precision=0.9, Recall=0.1
    - Arithmetic mean: (0.9 + 0.1)/2 = 0.5 (seems okay?)
    - Harmonic mean (F1): 0.18 (much lower, reflects poor recall!)
    
    This is good! If either precision or recall is low, F1 should be low too.
    
    THE INTUITION:
    -------------
    F1 score punishes extreme imbalances between precision and recall.
    
    Best case: Precision=1.0, Recall=1.0 → F1=1.0 (perfect!)
    Balanced: Precision=0.8, Recall=0.8 → F1=0.8 (good)
    Imbalanced: Precision=0.9, Recall=0.1 → F1=0.18 (bad!)
    
    WHEN TO USE F1:
    --------------
    - You care about both precision and recall equally
    - You want a single number for comparison
    - You have imbalanced classes (where accuracy is misleading)
    
    WHEN NOT TO USE:
    ---------------
    - You care more about precision than recall (or vice versa)
    - Use F-beta score instead for weighted tradeoff
    
    THE F-BETA FAMILY:
    -----------------
    F1 is part of a family: F-beta = (1+β²) × P×R / (β²×P + R)
    - β < 1: Favors precision (β=0.5 → F0.5)
    - β = 1: Balanced (F1, what we implement here)
    - β > 1: Favors recall (β=2 → F2)
    
    PARAMETERS:
    ----------
    y_true : Matrix
        True labels
        
    y_pred : Matrix
        Predicted labels
        
    positive_class : int
        Which class to treat as "positive"
        
    RETURNS:
    -------
    f1 : float
        F1 score between 0 and 1 (higher is better)
        
    EXAMPLE:
    -------
    >>> y_true = Matrix([[1], [0], [1], [0], [1], [1]])
    >>> y_pred = Matrix([[1], [0], [1], [1], [0], [1]])
    >>> 
    >>> p = precision(y_true, y_pred)  # 0.75 (3 correct out of 4 predictions)
    >>> r = recall(y_true, y_pred)     # 0.75 (3 found out of 4 actual)
    >>> f1 = f1_score(y_true, y_pred)  # 0.75 (balanced)
    """
    
    prec = precision(y_true, y_pred, positive_class)
    rec = recall(y_true, y_pred, positive_class)
    
    if prec + rec == 0:
        # Both precision and recall are 0
        return 0.0
    
    return 2 * prec * rec / (prec + rec)


def specificity(y_true: Matrix, y_pred: Matrix, positive_class: int = 1) -> float:
    """
    Specificity: Of all actual negatives, how many did we correctly identify?
    
    Also called "True Negative Rate" (TNR). This is the "recall for negatives" —
    it measures how good you are at identifying negative cases.
    
    THE FORMULA:
    -----------
    Specificity = TN / (TN + FP)
                = True Negatives / All Actual Negatives
    
    THE INTUITION:
    -------------
    High specificity means you rarely call something positive when it's actually
    negative — few false alarms.
    
    Example: Medical test with high specificity
    - If test says "no disease," it's almost always right
    - Healthy people rarely get false positive results
    
    SPECIFICITY VS SENSITIVITY (RECALL):
    -----------------------------------
    - Sensitivity (Recall): How many positives did you catch?
    - Specificity: How many negatives did you correctly reject?
    
    Both matter! A good classifier has high sensitivity AND high specificity.
    
    WHEN IT MATTERS:
    ---------------
    - You want to avoid false alarms
    - Correctly identifying negatives is important
    - Example: Drug safety — important to identify when drug is NOT safe
    
    PARAMETERS:
    ----------
    y_true : Matrix
        True labels
        
    y_pred : Matrix  
        Predicted labels
        
    positive_class : int
        Which class to treat as "positive"
        
    RETURNS:
    -------
    specificity : float
        Between 0 and 1 (higher is better)
    """
    
    cm = ConfusionMatrix(y_true, y_pred)
    tp, tn, fp, fn = cm.get_tp_tn_fp_fn(positive_class)
    
    if tn + fp == 0:
        # No actual negatives
        return 0.0
    
    return tn / (tn + fp)


# ==============================================================================
# REGRESSION METRICS
# ==============================================================================

def r_squared(y_true: Matrix, y_pred: Matrix) -> float:
    """
    R² (coefficient of determination) for regression.
    
    This measures how well your model explains the variance in the target variable.
    It's THE standard metric for regression!
    
    THE FORMULA:
    -----------
    R² = 1 - (SS_residual / SS_total)
    
    Where:
    - SS_residual = Σ(yᵢ - ŷᵢ)² (unexplained variance)
    - SS_total = Σ(yᵢ - ȳ)² (total variance)
    
    THE INTUITION:
    -------------
    R² answers: "What fraction of variance in y does my model explain?"
    
    - R² = 1.0: Perfect predictions! Model explains all variance.
    - R² = 0.8: Model explains 80% of variance (good!)
    - R² = 0.0: Model is no better than just predicting the mean
    - R² < 0.0: Model is WORSE than predicting the mean (very bad!)
    
    THE BASELINE COMPARISON:
    -----------------------
    R² compares your model to the dumbest possible model: always predict ȳ (mean).
    
    If you always predict ȳ:
    - SS_residual = SS_total (all variance unexplained)
    - R² = 1 - (SS_total / SS_total) = 0
    
    Your model should do better than this!
    
    INTERPRETING R²:
    ---------------
    - R² = 0.9: Excellent! Model captures 90% of variance
    - R² = 0.7-0.9: Good, captures most patterns
    - R² = 0.4-0.7: Moderate, some predictive power
    - R² = 0.0-0.4: Weak, barely better than mean
    - R² < 0.0: Model is actively harmful!
    
    WHEN R² CAN BE NEGATIVE:
    -----------------------
    If your model is very bad (predicts wildly wrong), it can be worse than
    just predicting the mean. Then SS_residual > SS_total, giving R² < 0.
    
    This is a red flag that something is seriously wrong!
    
    LIMITATIONS:
    -----------
    - R² always increases when you add more features (even random ones!)
    - Use adjusted R² for model comparison
    - Doesn't tell you if predictions are biased
    - Sensitive to outliers
    
    PARAMETERS:
    ----------
    y_true : Matrix, shape (n_samples, 1)
        True target values
        
    y_pred : Matrix, shape (n_samples, 1)
        Predicted target values
        
    RETURNS:
    -------
    r2 : float
        R² score (typically between 0 and 1, but can be negative)
        
    EXAMPLE:
    -------
    >>> y_true = Matrix([[1], [2], [3], [4], [5]])
    >>> y_pred = Matrix([[1.1], [2.1], [2.9], [4.2], [4.8]])
    >>> 
    >>> print(r_squared(y_true, y_pred))
    0.98  # Excellent predictions!
    """
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")
    
    n_samples = y_true.num_rows
    
    # Compute mean of y_true
    y_values = [y_true[i, 0] for i in range(n_samples)]
    y_mean = sum(y_values) / n_samples
    
    # Compute SS_residual (sum of squared residuals)
    ss_residual = sum((y_true[i, 0] - y_pred[i, 0]) ** 2 
                     for i in range(n_samples))
    
    # Compute SS_total (total sum of squares)
    ss_total = sum((y_true[i, 0] - y_mean) ** 2 
                  for i in range(n_samples))
    
    # Handle edge case: all y values are the same
    if ss_total < 1e-10:
        # No variance to explain
        if ss_residual < 1e-10:
            return 1.0  # Perfect predictions of constant value
        else:
            return 0.0  # Can't explain zero variance
    
    r2 = 1.0 - (ss_residual / ss_total)
    return r2


def mean_absolute_error(y_true: Matrix, y_pred: Matrix) -> float:
    """
    Mean Absolute Error (MAE): Average absolute difference between predictions and truth.
    
    THE FORMULA:
    -----------
    MAE = (1/n) Σ |yᵢ - ŷᵢ|
    
    THE INTUITION:
    -------------
    MAE tells you "on average, how far off are my predictions?" in the same units
    as your target variable.
    
    Example: If predicting house prices with MAE = $50,000, your predictions are
    off by an average of $50,000.
    
    MAE VS MSE:
    ----------
    - MAE: Average absolute error (treats all errors equally)
    - MSE: Average squared error (penalizes large errors more)
    
    MAE is more robust to outliers than MSE. If you have a few wild predictions,
    MAE won't be as affected.
    
    WHEN TO USE MAE:
    ---------------
    - You want interpretable errors in original units
    - Outliers exist and you don't want them dominating the metric
    - All errors should be weighted equally
    
    PARAMETERS:
    ----------
    y_true : Matrix
        True values
        
    y_pred : Matrix
        Predicted values
        
    RETURNS:
    -------
    mae : float
        Mean absolute error (same units as y)
    """
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")
    
    n_samples = y_true.num_rows
    
    total_error = sum(abs(y_true[i, 0] - y_pred[i, 0]) for i in range(n_samples))
    
    return total_error / n_samples


def root_mean_squared_error(y_true: Matrix, y_pred: Matrix) -> float:
    """
    Root Mean Squared Error (RMSE): Square root of average squared errors.
    
    THE FORMULA:
    -----------
    RMSE = sqrt((1/n) Σ (yᵢ - ŷᵢ)²)
    
    THE INTUITION:
    -------------
    RMSE is like MAE but penalizes large errors more heavily (because of squaring).
    
    RMSE vs MAE:
    - MAE = 10: Average error is 10 units
    - RMSE = 10: Typical error is 10 units, but a few large errors pull it up
    
    If RMSE >> MAE, you have some large errors (outliers).
    If RMSE ≈ MAE, errors are roughly uniformly distributed.
    
    WHEN TO USE RMSE:
    ----------------
    - Large errors are especially bad (you want to penalize them heavily)
    - Standard choice in many competitions
    - Continuous optimization (differentiable)
    
    PARAMETERS:
    ----------
    y_true : Matrix
        True values
        
    y_pred : Matrix
        Predicted values
        
    RETURNS:
    -------
    rmse : float
        Root mean squared error
    """
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")
    
    n_samples = y_true.num_rows
    
    squared_errors = sum((y_true[i, 0] - y_pred[i, 0]) ** 2 
                        for i in range(n_samples))
    
    mse = squared_errors / n_samples
    rmse = mse ** 0.5
    
    return rmse


# ==============================================================================
# CLASSIFICATION REPORT
# ==============================================================================

def classification_report(
    y_true: Matrix,
    y_pred: Matrix,
    class_names: Optional[List[str]] = None
) -> str:
    """
    Generate a comprehensive classification report.
    
    This gives you ALL the important metrics in one convenient summary! It's
    like getting a report card for your classifier.
    
    THE REPORT INCLUDES:
    -------------------
    For each class:
    - Precision: How often predictions for this class are correct
    - Recall: How many samples of this class were found
    - F1-score: Balance of precision and recall
    - Support: Number of samples of this class
    
    Plus overall metrics:
    - Accuracy: Overall correctness
    - Macro average: Average metrics across classes (treats all classes equally)
    - Weighted average: Average weighted by class size
    
    MACRO VS WEIGHTED AVERAGE:
    -------------------------
    Macro average: Simple average across classes
    - Treats all classes equally, regardless of size
    - Good for balanced datasets
    
    Weighted average: Average weighted by number of samples per class
    - Gives more importance to larger classes
    - Good for imbalanced datasets
    
    PARAMETERS:
    ----------
    y_true : Matrix
        True labels
        
    y_pred : Matrix
        Predicted labels
        
    class_names : List[str], optional
        Names for each class (for readability)
        
    RETURNS:
    -------
    report : str
        Formatted classification report
        
    EXAMPLE:
    -------
    >>> y_true = Matrix([[0], [1], [2], [0], [1], [2]])
    >>> y_pred = Matrix([[0], [1], [2], [0], [2], [2]])
    >>> 
    >>> print(classification_report(y_true, y_pred, 
    ...                            class_names=['cat', 'dog', 'bird']))
    """
    
    cm = ConfusionMatrix(y_true, y_pred)
    n_classes = cm.n_classes
    
    # Default class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    if len(class_names) != n_classes:
        raise ValueError(
            f"class_names has {len(class_names)} names but data has {n_classes} classes"
        )
    
    # Compute metrics for each class
    class_metrics = []
    for i in range(n_classes):
        prec = precision(y_true, y_pred, positive_class=i)
        rec = recall(y_true, y_pred, positive_class=i)
        f1 = f1_score(y_true, y_pred, positive_class=i)
        
        # Support: number of samples of this class
        support = sum(1 for j in range(y_true.num_rows) if int(y_true[j, 0]) == i)
        
        class_metrics.append({
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'support': support
        })
    
    # Compute overall accuracy
    acc = accuracy(y_true, y_pred)
    
    # Compute macro averages (simple average)
    macro_precision = sum(m['precision'] for m in class_metrics) / n_classes
    macro_recall = sum(m['recall'] for m in class_metrics) / n_classes
    macro_f1 = sum(m['f1'] for m in class_metrics) / n_classes
    
    # Compute weighted averages (weighted by support)
    total_support = sum(m['support'] for m in class_metrics)
    weighted_precision = sum(m['precision'] * m['support'] for m in class_metrics) / total_support
    weighted_recall = sum(m['recall'] * m['support'] for m in class_metrics) / total_support
    weighted_f1 = sum(m['f1'] * m['support'] for m in class_metrics) / total_support
    
    # Format the report
    lines = []
    lines.append("=" * 70)
    lines.append("CLASSIFICATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    lines.append("-" * 70)
    
    # Per-class metrics
    for i, name in enumerate(class_names):
        m = class_metrics[i]
        lines.append(
            f"{name:<15} {m['precision']:>10.3f} {m['recall']:>10.3f} "
            f"{m['f1']:>10.3f} {m['support']:>10d}"
        )
    
    lines.append("-" * 70)
    
    # Overall accuracy
    lines.append(f"{'Accuracy':<15} {'':<10} {'':<10} {acc:>10.3f} {total_support:>10d}")
    lines.append("")
    
    # Macro average
    lines.append(
        f"{'Macro avg':<15} {macro_precision:>10.3f} {macro_recall:>10.3f} "
        f"{macro_f1:>10.3f} {total_support:>10d}"
    )
    
    # Weighted average
    lines.append(
        f"{'Weighted avg':<15} {weighted_precision:>10.3f} {weighted_recall:>10.3f} "
        f"{weighted_f1:>10.3f} {total_support:>10d}"
    )
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


# ==============================================================================
# EXAMPLE USAGE AND TESTING
# ==============================================================================

if __name__ == "__main__":
    """
    Demonstrate all metrics with examples.
    """
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        METRICS: MEASURING SUCCESS IN MACHINE LEARNING           ║
║                                                                  ║
║  "My model is 99% accurate!" - Me, before learning about       ║
║   precision, recall, and the curse of imbalanced data           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # =========================================================================
    # Demo 1: Binary Classification - Balanced Dataset
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("DEMO 1: BINARY CLASSIFICATION (Balanced Dataset)")
    print("=" * 70)
    
    print("\nScenario: Email classification (spam vs not-spam)")
    print("Dataset: 50% spam, 50% not-spam (balanced)")
    
    y_true_balanced = Matrix([[1], [0], [1], [1], [0], [1], [0], [0]])
    y_pred_balanced = Matrix([[1], [0], [1], [0], [0], [1], [1], [0]])
    
    print("\nTrue labels:", [int(y_true_balanced[i,0]) for i in range(8)])
    print("Predictions:", [int(y_pred_balanced[i,0]) for i in range(8)])
    
    # Confusion matrix
    cm = ConfusionMatrix(y_true_balanced, y_pred_balanced)
    print("\n" + str(cm))
    
    # Metrics
    acc = accuracy(y_true_balanced, y_pred_balanced)
    prec = precision(y_true_balanced, y_pred_balanced, positive_class=1)
    rec = recall(y_true_balanced, y_pred_balanced, positive_class=1)
    f1 = f1_score(y_true_balanced, y_pred_balanced, positive_class=1)
    
    print(f"\nAccuracy:  {acc:.3f} (overall correctness)")
    print(f"Precision: {prec:.3f} (of predicted spam, how many are actually spam?)")
    print(f"Recall:    {rec:.3f} (of actual spam, how many did we catch?)")
    print(f"F1-Score:  {f1:.3f} (balance of precision and recall)")
    
    print("\nFor balanced datasets, accuracy is a reasonable metric!")
    
    # =========================================================================
    # Demo 2: Imbalanced Dataset - The Accuracy Paradox
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("DEMO 2: THE ACCURACY PARADOX (Imbalanced Dataset)")
    print("=" * 70)
    
    print("\nScenario: Fraud detection")
    print("Dataset: 95% legitimate, 5% fraud (highly imbalanced!)")
    
    # 20 transactions: 19 legitimate, 1 fraud
    y_true_imbalanced = Matrix([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0],
                                [0], [0], [0], [0], [0], [0], [0], [0], [0], [1]])
    
    # Useless model: always predicts "not fraud"
    y_pred_useless = Matrix([[0]] * 20)
    
    # Smart model: catches the fraud but has 2 false alarms
    y_pred_smart = Matrix([[0], [0], [0], [0], [0], [0], [0], [0], [1], [0],
                           [0], [0], [1], [0], [0], [0], [0], [0], [0], [1]])
    
    print("\nUseless model (always predicts 'not fraud'):")
    acc_useless = accuracy(y_true_imbalanced, y_pred_useless)
    rec_useless = recall(y_true_imbalanced, y_pred_useless, positive_class=1)
    print(f"  Accuracy: {acc_useless:.1%} (looks great!)")
    print(f"  Recall:   {rec_useless:.1%} (but catches 0% of fraud!)")
    
    print("\nSmart model (catches fraud with some false alarms):")
    acc_smart = accuracy(y_true_imbalanced, y_pred_smart)
    prec_smart = precision(y_true_imbalanced, y_pred_smart, positive_class=1)
    rec_smart = recall(y_true_imbalanced, y_pred_smart, positive_class=1)
    f1_smart = f1_score(y_true_imbalanced, y_pred_smart, positive_class=1)
    
    print(f"  Accuracy:  {acc_smart:.1%} (slightly lower)")
    print(f"  Precision: {prec_smart:.1%} (1 correct out of 3 predictions)")
    print(f"  Recall:    {rec_smart:.1%} (caught the fraud!)")
    print(f"  F1-Score:  {f1_smart:.3f}")
    
    print("\n→ Smart model has LOWER accuracy but is much more useful!")
    print("  This is the accuracy paradox with imbalanced data.")
    
    # =========================================================================
    # Demo 3: Multi-Class Classification
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("DEMO 3: MULTI-CLASS CLASSIFICATION")
    print("=" * 70)
    
    print("\nScenario: Image classification (cat/dog/bird)")
    
    y_true_multi = Matrix([
        [0], [1], [2], [0], [1], [2], [0], [1], [2],  # One of each class per row
        [0], [1], [2]
    ])
    
    y_pred_multi = Matrix([
        [0], [1], [2], [0], [2], [2], [1], [1], [2],  # Some errors
        [0], [1], [1]  # Confused bird for dog
    ])
    
    # Classification report
    report = classification_report(
        y_true_multi, y_pred_multi,
        class_names=['Cat', 'Dog', 'Bird']
    )
    print("\n" + report)
    
    print("\nInterpretation:")
    print("- Cat: High precision (when we say cat, usually right)")
    print("- Dog: Lower recall (missed some dogs)")
    print("- Bird: Worst performance (confused with dog)")
    
    # =========================================================================
    # Demo 4: Regression Metrics
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("DEMO 4: REGRESSION METRICS")
    print("=" * 70)
    
    print("\nScenario: House price prediction")
    
    y_true_reg = Matrix([[300], [450], [500], [350], [600]])
    y_pred_reg = Matrix([[310], [440], [520], [360], [590]])
    
    print("\nTrue prices (in $1000s):", 
          [int(y_true_reg[i,0]) for i in range(5)])
    print("Predicted prices:", 
          [int(y_pred_reg[i,0]) for i in range(5)])
    
    r2 = r_squared(y_true_reg, y_pred_reg)
    mae = mean_absolute_error(y_true_reg, y_pred_reg)
    rmse = root_mean_squared_error(y_true_reg, y_pred_reg)
    
    print(f"\nR² Score: {r2:.4f} (explains {r2*100:.1f}% of variance)")
    print(f"MAE: ${mae:.1f}K (average error)")
    print(f"RMSE: ${rmse:.1f}K (typical error, penalizes large errors)")
    
    print("\n✓ All metrics show good predictions (close to true values)")
    
    # =========================================================================
    # Key Takeaways
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. ACCURACY: Simple but misleading with imbalanced data
   - Use for balanced datasets
   - Don't trust it alone for imbalanced classes!

2. PRECISION: When you predict positive, how often are you right?
   - High precision = Few false alarms
   - Important when false positives are costly

3. RECALL: Of all actual positives, how many did you find?
   - High recall = Catch most/all positive cases
   - Important when false negatives are costly

4. F1 SCORE: Balance of precision and recall
   - Use when you care about both equally
   - Harmonic mean punishes extreme imbalances

5. CONFUSION MATRIX: Shows the full picture
   - TP, TN, FP, FN tell you everything
   - All other metrics derive from these

6. FOR IMBALANCED DATA:
   - Ignore accuracy!
   - Focus on precision, recall, F1
   - Consider using different thresholds
   - Maybe use class weights during training

7. FOR REGRESSION:
   - R²: How much variance explained (0 to 1)
   - MAE: Average error (same units as target)
   - RMSE: Typical error (penalizes outliers more)

8. PICK THE RIGHT METRIC FOR YOUR PROBLEM:
   - Medical diagnosis: High recall (catch all diseases)
   - Spam filter: High precision (don't block real emails)
   - Fraud detection: Balance with F1 score
   - House prices: R² and RMSE

The golden rule: Always understand your problem before choosing metrics!
Different applications need different tradeoffs.
    """)