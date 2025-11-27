"""
REGULARIZATION HELPERS: Cross-Validation and Model Selection
============================================================

This module provides utilities for choosing regularization hyperparameters and 
evaluating regularized models. The most important function here is cross-validation, 
which is THE proper way to choose regularization strength.

MY LEARNING JOURNEY:
-------------------
Early mistake: I used to choose α by training on the training set and evaluating 
on the training set. I'd always pick α=0 (no regularization) because that gave 
the best training error. Then I'd test on test data and get terrible results.

The problem: I was cheating! By choosing α based on training performance, I was 
essentially using the training data twice: once to fit weights, again to choose α. 
This defeats the purpose of having a test set.

The solution: Cross-validation. Split training data into K folds, train on K-1 
folds, validate on the remaining fold, repeat K times, average results. Choose 
the α that gives the best average validation performance.

This gives an honest estimate of how well each α will generalize to truly unseen data!

THE CROSS-VALIDATION ALGORITHM:
-------------------------------
K-Fold Cross-Validation:

1. Split data into K equal folds (typically K=5 or K=10)
2. For each α value you want to try:
   For fold i in 1..K:
       - Train on all folds except i
       - Validate on fold i
       - Record validation score
   Average the K validation scores
3. Pick the α with best average validation score
4. Retrain on ALL training data with that α
5. Finally evaluate on test set (only once!)

This is the gold standard for hyperparameter tuning.

WHY K-FOLD INSTEAD OF SINGLE VALIDATION SET?
--------------------------------------------
You could just split into train/validation/test (e.g., 60/20/20). This works but:

1. Uses less data for training (only 60% instead of 80% in 5-fold CV)
2. Single validation set might be unrepresentative (bad luck)
3. K-fold gives more stable estimates (averages over K different splits)

K-fold cross-validation is more robust and uses data more efficiently.

COMMON VALUES OF K:
------------------
K=5: Common default, good balance of computation and reliability
K=10: More stable estimates, but 2x slower than K=5
K=n (leave-one-out): Maximum data usage, very slow for large datasets
K=3: Faster but less stable, okay for quick experiments

I typically use K=5 unless I have very little data (then K=10) or a lot of 
data (then K=3 for speed).

THE REGULARIZATION PATH:
------------------------
Another useful visualization is the "regularization path": plot how each weight 
changes as you vary α from 0 to large values.

For Ridge: All weights smoothly shrink toward zero as α increases
For Lasso: Weights drop to zero one by one as α increases
For Elastic Net: Combines both behaviors

This helps you understand:
- Which features are most important (last to be eliminated)
- Whether you're in the over-regularized regime (all weights near zero)
- How stable feature selection is across α values

GRID SEARCH VS RANDOM SEARCH:
-----------------------------
Grid Search: Try all combinations of hyperparameters on a grid
- Systematic and thorough
- Guarantees you won't miss good combinations
- Can be slow with many hyperparameters

Random Search: Try random combinations
- Often finds good solutions faster
- Better for high-dimensional hyperparameter spaces
- Might miss the global optimum

For regularization, grid search works well because we typically only tune 1-2 
parameters (α for Ridge/Lasso, or α and l1_ratio for Elastic Net).

Author: Mrigesh (learning that choosing hyperparameters is as important as choosing the algorithm)
"""


import random
from typing import Dict, List, Optional, Tuple

from foundations.linear_algebra.vectors_and_matrices import Matrix


def k_fold_split(X: Matrix, y: Matrix, k: int = 5, shuffle: bool = True, 
                 random_state: Optional[int] = None) -> List[Tuple[Matrix, Matrix, Matrix, Matrix]]:
    """
    Split data into K folds for cross-validation.
    
    This creates K different train/validation splits, where each fold gets to be 
    the validation set once. This ensures every data point is used for validation 
    exactly once, maximizing data efficiency.
    
    THE ALGORITHM:
    -------------
    1. Optionally shuffle the data (recommended!)
    2. Divide into K approximately equal-sized folds
    3. For each fold i:
       - Use fold i as validation
       - Use all other folds as training
       - Yield (X_train, X_val, y_train, y_val)
    
    EXAMPLE WITH K=5:
    ----------------
    Data: [A, B, C, D, E, F, G, H, I, J] (10 samples)
    
    Fold 1: Train on [C,D,E,F,G,H,I,J], Validate on [A,B]
    Fold 2: Train on [A,B,E,F,G,H,I,J], Validate on [C,D]
    Fold 3: Train on [A,B,C,D,G,H,I,J], Validate on [E,F]
    Fold 4: Train on [A,B,C,D,E,F,I,J], Validate on [G,H]
    Fold 5: Train on [A,B,C,D,E,F,G,H], Validate on [I,J]
    
    Every sample appears in validation exactly once!
    
    PARAMETERS:
    ----------
    X : Matrix
        Features
        
    y : Matrix
        Targets
        
    k : int, default=5
        Number of folds
        
    shuffle : bool, default=True
        Whether to shuffle before splitting
        Recommended: True (unless you have time series data)
        
    random_state : int or None
        Random seed for reproducibility
        
    RETURNS:
    -------
    folds : List[Tuple[Matrix, Matrix, Matrix, Matrix]]
        List of (X_train, X_val, y_train, y_val) for each fold
    """
    
    if X.num_rows != y.num_rows:
        raise ValueError("X and y must have same number of samples")
    
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    
    if k > X.num_rows:
        raise ValueError(f"k={k} is larger than number of samples ({X.num_rows})")
    
    n_samples = X.num_rows
    
    # Create indices
    indices = list(range(n_samples))
    
    # Shuffle if requested
    if shuffle:
        if random_state is not None:
            random.seed(random_state)
        random.shuffle(indices)
    
    # Compute fold sizes (handle case where n_samples not divisible by k)
    fold_sizes = [n_samples // k] * k
    for i in range(n_samples % k):
        fold_sizes[i] += 1
    
    # Create folds
    folds = []
    current = 0
    
    for fold_idx in range(k):
        # Indices for validation fold
        val_start = current
        val_end = current + fold_sizes[fold_idx]
        val_indices = indices[val_start:val_end]
        
        # Indices for training (everything except validation fold)
        train_indices = indices[:val_start] + indices[val_end:]
        
        # Extract data
        X_train_data = [X.elements[i] for i in train_indices]
        X_val_data = [X.elements[i] for i in val_indices]
        y_train_data = [y.elements[i] for i in train_indices]
        y_val_data = [y.elements[i] for i in val_indices]
        
        X_train = Matrix(X_train_data)
        X_val = Matrix(X_val_data)
        y_train = Matrix(y_train_data)
        y_val = Matrix(y_val_data)
        
        folds.append((X_train, X_val, y_train, y_val))
        
        current = val_end
    
    return folds


def cross_validate_ridge(
    X: Matrix, 
    y: Matrix, 
    alphas: List[float],
    k_folds: int = 5,
    n_epochs: int = 200,
    verbose: bool = False
) -> Dict:
    """
    Perform K-fold cross-validation for Ridge regression across multiple α values.
    
    This function answers the question: "Which α should I use for Ridge regression?"
    
    It trains a Ridge model for each α value using K-fold cross-validation and 
    returns the validation scores. You can then pick the α with the best average 
    validation performance.
    
    THE PROCESS:
    -----------
    For each α in alphas:
        For each fold in K folds:
            1. Train Ridge model with this α on training folds
            2. Evaluate on validation fold
            3. Record R² score
        Average the K validation scores
    
    Return all scores so you can see the full picture!
    
    WHY THIS IS THE RIGHT WAY:
    -------------------------
    This gives an honest estimate of generalization performance. We're not 
    choosing α based on test data (that would be cheating!), and we're not 
    choosing based on training data (that would favor α=0).
    
    Instead, we're choosing based on data the model hasn't been trained on 
    (validation folds), which simulates how well the model will do on truly 
    unseen test data.
    
    PARAMETERS:
    ----------
    X : Matrix
        Features (should already be standardized and have bias column!)
        
    y : Matrix
        Targets
        
    alphas : List[float]
        List of α values to try
        Example: [0.001, 0.01, 0.1, 1.0, 10.0]
        
    k_folds : int, default=5
        Number of cross-validation folds
        
    n_epochs : int, default=200
        Number of training epochs per model
        
    verbose : bool, default=False
        Whether to print progress
        
    RETURNS:
    -------
    results : dict
        Dictionary containing:
        - 'alphas': The α values tested
        - 'mean_scores': Mean validation R² for each α
        - 'std_scores': Standard deviation of validation R² for each α
        - 'fold_scores': All K scores for each α (for detailed analysis)
        - 'best_alpha': The α with highest mean validation score
        - 'best_score': The best mean validation score
    
    EXAMPLE:
    -------
    >>> from ridge_regression import RidgeRegression
    >>> 
    >>> alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    >>> results = cross_validate_ridge(X_train, y_train, alphas, k_folds=5)
    >>> 
    >>> print(f"Best α: {results['best_alpha']}")
    >>> print(f"Best validation R²: {results['best_score']:.4f}")
    >>> 
    >>> # Train final model with best α
    >>> final_model = RidgeRegression(alpha=results['best_alpha'])
    >>> final_model.fit(X_train, y_train)
    """
    
    from ridge_regression import RidgeRegression
    
    if verbose:
        print("=" * 70)
        print("RIDGE REGRESSION CROSS-VALIDATION")
        print("=" * 70)
        print(f"Testing {len(alphas)} α values with {k_folds}-fold CV")
        print()
    
    # Create folds
    folds = k_fold_split(X, y, k=k_folds, shuffle=True, random_state=42)
    
    # Store results
    all_scores = []  # List of lists: one list of K scores per α
    
    # Try each α value
    for alpha in alphas:
        fold_scores = []
        
        if verbose:
            print(f"Testing α = {alpha:8.4f}...", end=" ")
        
        # Evaluate on each fold
        for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(folds):
            # Train model
            model = RidgeRegression(
                alpha=alpha,
                learning_rate=0.01,
                n_epochs=n_epochs,
                verbose=False
            )
            model.fit(X_train, y_train)
            
            # Evaluate on validation fold
            val_score = model.score(X_val, y_val)
            fold_scores.append(val_score)
        
        all_scores.append(fold_scores)
        
        # Compute statistics
        mean_score = sum(fold_scores) / len(fold_scores)
        variance = sum((s - mean_score) ** 2 for s in fold_scores) / len(fold_scores)
        std_score = variance ** 0.5
        
        if verbose:
            print(f"Mean R² = {mean_score:.4f} (±{std_score:.4f})")
    
    # Compute mean scores for each α
    mean_scores = [sum(scores) / len(scores) for scores in all_scores]
    
    # Compute std scores for each α
    std_scores = []
    for scores in all_scores:
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std_scores.append(variance ** 0.5)
    
    # Find best α
    best_idx = mean_scores.index(max(mean_scores))
    best_alpha = alphas[best_idx]
    best_score = mean_scores[best_idx]
    
    if verbose:
        print()
        print("-" * 70)
        print(f"BEST: α = {best_alpha} with mean R² = {best_score:.4f}")
        print("=" * 70)
    
    return {
        'alphas': alphas,
        'mean_scores': mean_scores,
        'std_scores': std_scores,
        'fold_scores': all_scores,
        'best_alpha': best_alpha,
        'best_score': best_score
    }


def cross_validate_lasso(
    X: Matrix,
    y: Matrix,
    alphas: List[float],
    k_folds: int = 5,
    n_epochs: int = 300,
    verbose: bool = False
) -> Dict:
    """
    Perform K-fold cross-validation for Lasso regression.
    
    Same as cross_validate_ridge but for Lasso. Lasso typically needs more 
    epochs to converge due to coordinate descent.
    """
    
    from lasso_regression import LassoRegression
    
    if verbose:
        print("=" * 70)
        print("LASSO REGRESSION CROSS-VALIDATION")
        print("=" * 70)
        print(f"Testing {len(alphas)} α values with {k_folds}-fold CV")
        print()
    
    folds = k_fold_split(X, y, k=k_folds, shuffle=True, random_state=42)
    all_scores = []
    
    for alpha in alphas:
        fold_scores = []
        
        if verbose:
            print(f"Testing α = {alpha:8.4f}...", end=" ")
        
        for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(folds):
            model = LassoRegression(
                alpha=alpha,
                n_epochs=n_epochs,
                verbose=False
            )
            model.fit(X_train, y_train)
            val_score = model.score(X_val, y_val)
            fold_scores.append(val_score)
        
        all_scores.append(fold_scores)
        mean_score = sum(fold_scores) / len(fold_scores)
        variance = sum((s - mean_score) ** 2 for s in fold_scores) / len(fold_scores)
        std_score = variance ** 0.5
        
        if verbose:
            print(f"Mean R² = {mean_score:.4f} (±{std_score:.4f})")
    
    mean_scores = [sum(scores) / len(scores) for scores in all_scores]
    std_scores = []
    for scores in all_scores:
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std_scores.append(variance ** 0.5)
    
    best_idx = mean_scores.index(max(mean_scores))
    best_alpha = alphas[best_idx]
    best_score = mean_scores[best_idx]
    
    if verbose:
        print()
        print("-" * 70)
        print(f"BEST: α = {best_alpha} with mean R² = {best_score:.4f}")
        print("=" * 70)
    
    return {
        'alphas': alphas,
        'mean_scores': mean_scores,
        'std_scores': std_scores,
        'fold_scores': all_scores,
        'best_alpha': best_alpha,
        'best_score': best_score
    }


def cross_validate_elastic_net(
    X: Matrix,
    y: Matrix,
    alphas: List[float],
    l1_ratios: List[float] = [0.5],
    k_folds: int = 5,
    n_epochs: int = 300,
    verbose: bool = False
) -> Dict:
    """
    Perform K-fold cross-validation for Elastic Net.
    
    This is slightly more complex because Elastic Net has two hyperparameters: 
    α (overall strength) and l1_ratio (L1/L2 mix). We try all combinations!
    
    GRID SEARCH OVER TWO PARAMETERS:
    --------------------------------
    For each α:
        For each l1_ratio:
            Perform K-fold CV
            Record average validation score
    
    Pick the (α, l1_ratio) combination with best average score.
    
    PARAMETERS:
    ----------
    X, y : Matrix
        Data
        
    alphas : List[float]
        Overall regularization strengths to try
        
    l1_ratios : List[float], default=[0.5]
        L1/L2 mix ratios to try
        Default tries only balanced mix (0.5)
        Try [0.1, 0.5, 0.9] to explore the full spectrum
        
    k_folds, n_epochs, verbose : same as other CV functions
    
    RETURNS:
    -------
    results : dict
        Contains best_alpha, best_l1_ratio, best_score, and full grid results
    """
    
    from elastic_net import ElasticNet
    
    if verbose:
        print("=" * 70)
        print("ELASTIC NET CROSS-VALIDATION")
        print("=" * 70)
        print(f"Testing {len(alphas)} α values × {len(l1_ratios)} l1_ratios")
        print(f"Total: {len(alphas) * len(l1_ratios)} combinations with {k_folds}-fold CV")
        print()
    
    folds = k_fold_split(X, y, k=k_folds, shuffle=True, random_state=42)
    
    # Store results for all combinations
    grid_results = []
    
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            fold_scores = []
            
            if verbose:
                print(f"Testing α={alpha:7.4f}, l1_ratio={l1_ratio:.2f}...", end=" ")
            
            for X_train, X_val, y_train, y_val in folds:
                model = ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    n_epochs=n_epochs,
                    verbose=False
                )
                model.fit(X_train, y_train)
                val_score = model.score(X_val, y_val)
                fold_scores.append(val_score)
            
            mean_score = sum(fold_scores) / len(fold_scores)
            variance = sum((s - mean_score) ** 2 for s in fold_scores) / len(fold_scores)
            std_score = variance ** 0.5
            
            grid_results.append({
                'alpha': alpha,
                'l1_ratio': l1_ratio,
                'mean_score': mean_score,
                'std_score': std_score,
                'fold_scores': fold_scores
            })
            
            if verbose:
                print(f"Mean R² = {mean_score:.4f} (±{std_score:.4f})")
    
    # Find best combination
    best_result = max(grid_results, key=lambda x: x['mean_score'])
    
    if verbose:
        print()
        print("-" * 70)
        print(f"BEST: α={best_result['alpha']}, l1_ratio={best_result['l1_ratio']}")
        print(f"      Mean R² = {best_result['mean_score']:.4f}")
        print("=" * 70)
    
    return {
        'grid_results': grid_results,
        'best_alpha': best_result['alpha'],
        'best_l1_ratio': best_result['l1_ratio'],
        'best_score': best_result['mean_score']
    }


def plot_regularization_path(
    X: Matrix,
    y: Matrix,
    alphas: List[float],
    model_type: str = 'ridge'
) -> Dict:
    """
    Compute the regularization path: how weights change as α varies.
    
    This creates a visualization showing how each weight evolves as you increase 
    regularization strength. It's incredibly informative!
    
    WHAT YOU'LL SEE:
    ---------------
    Ridge: All weights smoothly shrink toward zero
    - No weights hit exactly zero
    - Important features shrink slower
    - Less important features shrink faster
    
    Lasso: Weights drop to zero one by one
    - Clear sparsity pattern
    - Can identify order of importance (last to drop = most important)
    - Some weights jump between positive/negative before zeroing
    
    Elastic Net: Combination of both behaviors
    - Smoother than Lasso
    - Still achieves sparsity
    - More stable selections
    
    INTERPRETATION:
    --------------
    - Features whose weights stay large across many α values are important
    - Features whose weights quickly drop to zero are less important
    - Sudden jumps might indicate correlated features (one takes over from another)
    
    PARAMETERS:
    ----------
    X, y : Matrix
        Training data
        
    alphas : List[float]
        α values to try (should be sorted from small to large)
        Example: [0.001, 0.01, 0.1, 1, 10, 100]
        
    model_type : str
        'ridge', 'lasso', or 'elastic_net'
        
    RETURNS:
    -------
    results : dict
        Contains 'alphas' and 'weights' (list of weight vectors, one per α)
    """
    
    if model_type == 'ridge':
        from ridge_regression import RidgeRegression
        ModelClass = RidgeRegression
        n_epochs = 200
    elif model_type == 'lasso':
        from lasso_regression import LassoRegression
        ModelClass = LassoRegression
        n_epochs = 300
    elif model_type == 'elastic_net':
        from elastic_net import ElasticNet
        ModelClass = ElasticNet
        n_epochs = 300
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Train model for each α and record weights
    weight_paths = []
    
    for alpha in alphas:
        if model_type == 'elastic_net':
            model = ModelClass(alpha=alpha, l1_ratio=0.5, n_epochs=n_epochs, verbose=False)
        else:
            model = ModelClass(alpha=alpha, n_epochs=n_epochs, verbose=False)
        
        model.fit(X, y)
        weight_paths.append(list(model.weights_.elements))
    
    return {
        'alphas': alphas,
        'weights': weight_paths  # weights[i] is weight vector for alphas[i]
    }