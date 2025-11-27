# Logistic Regression: From Predicting Numbers to Predicting Categories

> "Wait, it's called 'regression' but it does classification? That confused me for weeks until I understood it's really about modeling probabilities." — Me, discovering the beautiful trick at the heart of logistic regression

---

## What This Folder Contains

This is where I made the leap from regression (predicting continuous values) to classification (predicting discrete categories). Logistic regression is the foundation of all classification in machine learning, and understanding it deeply unlocked everything that came after.

I implemented three key components here:

1. **`logistic_regression_scratch.py`**: The full implementation using gradient descent with sigmoid activation
2. **`logistic_regression_evaluation.py`**: Classification-specific metrics and model evaluation
3. **`decision_boundary_visualization.py`**: Tools to visualize how the model separates classes

## The Core Problem: Predicting Categories

**The setup:** You have data points with features X, and each belongs to one of several categories. You want to predict which category a new point belongs to.

**Examples:**

- Email spam detection: Given email features, predict spam (1) or not spam (0)
- Medical diagnosis: Given patient symptoms, predict disease (1) or healthy (0)
- Customer churn: Given usage patterns, predict will leave (1) or stay (0)

Unlike linear regression where we predict any real number, here we need to predict discrete categories. But here's the beautiful insight: instead of directly predicting categories, we predict the probability of each category, then pick the most likely one!

## The Key Insight: From Linear to Probabilistic

When I first tried classification, my naive approach was to use linear regression and round the output. If the model predicted 0.7, I'd call it class 1. If it predicted 0.3, I'd call it class 0. This seemed reasonable, but it had catastrophic problems!

**The problems with using linear regression for classification:**

1. **Unbounded outputs**: Linear regression can predict any value (negative infinity to positive infinity). But probabilities must be between 0 and 1!

2. **No probabilistic interpretation**: What does it mean when linear regression predicts 2.5 for a binary classification problem? Or negative 0.8? These aren't probabilities.

3. **Sensitive to outliers**: A single extreme outlier can completely destroy the decision boundary. Imagine one spam email with unusual features getting a large weight, pulling the entire line in the wrong direction.

4. **Wrong loss function**: Mean squared error isn't the right way to measure classification performance. We need something designed for probabilities.

Then I discovered the elegant solution: transform the linear output into a probability using the sigmoid function! This simple transformation changes everything.

## The Sigmoid Transformation: The Heart of Logistic Regression

The sigmoid function (also called the logistic function, which is where "logistic regression" gets its name) is the magic that makes everything work. Here's what it does:

```
σ(z) = 1 / (1 + e^(-z))
```

This function takes any real number z and squashes it into the range (0, 1). Let me show you why this is perfect for classification:

**Properties of sigmoid:**

- Input -∞ → Output approaches 0
- Input 0 → Output exactly 0.5
- Input +∞ → Output approaches 1
- Always between 0 and 1 (perfect for probabilities!)
- Smooth and differentiable (we can use gradient descent!)

**The intuition:** Think of z as a "confidence score" from your linear model. Large positive z means very confident it's class 1. Large negative z means very confident it's class 0. The sigmoid converts this unbounded confidence into a bounded probability.

**Concrete example:**

```
z = -5 → σ(z) = 0.007 → About 1% probability of class 1 (very confident it's class 0)
z = 0  → σ(z) = 0.5   → Uncertain, 50-50 split
z = +5 → σ(z) = 0.993 → About 99% probability of class 1 (very confident it's class 1)
```

## The Complete Logistic Regression Model

Putting it all together, logistic regression works in two steps:

**Step 1: Compute linear combination (just like linear regression)**

```
z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = w^T x
```

This gives an unbounded score based on the features.

**Step 2: Apply sigmoid to get probability**

```
P(y=1|x) = σ(z) = 1 / (1 + e^(-w^T x))
```

This transforms the score into a probability between 0 and 1.

**Making predictions:**

- If P(y=1|x) ≥ 0.5, predict class 1
- If P(y=1|x) < 0.5, predict class 0

The decision boundary is where P(y=1|x) = 0.5, which happens when w^T x = 0. This is a hyperplane in feature space, just like linear regression! The difference is how we interpret values on either side of that plane.

## 📐 The Mathematics: Why Cross-Entropy Loss?

When I was implementing this, I initially tried using mean squared error (MSE) like in linear regression. The model trained, but convergence was painfully slow and the gradients behaved strangely. Then I learned about cross-entropy loss and everything clicked.

**The intuition behind cross-entropy:**

Cross-entropy measures how "surprised" we are by the true label given our predicted probability. If we predict high probability for the correct class, surprise is low (good!). If we predict low probability for the correct class, surprise is high (bad!).

**The formula for binary classification:**

```
Loss = -(1/n) Σ [y_i × log(ŷ_i) + (1-y_i) × log(1-ŷ_i)]
```

Let me break down what this means:

- When the true label is 1 (y=1): Loss = -log(ŷ)

  - If we predict ŷ=0.9 (confident correct): Loss = -log(0.9) = 0.105 (low)
  - If we predict ŷ=0.1 (confident wrong): Loss = -log(0.1) = 2.303 (high)

- When the true label is 0 (y=0): Loss = -log(1-ŷ)
  - If we predict ŷ=0.1 (confident correct): Loss = -log(0.9) = 0.105 (low)
  - If we predict ŷ=0.9 (confident wrong): Loss = -log(0.1) = 2.303 (high)

The logarithm creates a penalty that grows exponentially as we become more confident in the wrong answer. This strongly discourages overconfident mistakes!

**Why not MSE for classification?**

The beautiful thing about using sigmoid with cross-entropy is the gradient. After all the calculus, the gradient simplifies to:

```
∂Loss/∂w = (1/n) X^T (ŷ - y)
```

This is identical to linear regression's gradient! Just the prediction error weighted by features. This clean gradient makes training fast and stable.

With MSE, the gradient involves the sigmoid derivative σ'(z) = σ(z)(1-σ(z)), which approaches zero when the model is very confident (even if wrong!). This causes the "vanishing gradient" problem where the model gets stuck and stops learning. Cross-entropy doesn't have this problem because the sigmoid derivative cancels out in the gradient formula.

## Decision Boundaries: Visualizing What the Model Learns

One of the most illuminating things I did was visualize the decision boundary. This is the line (or hyperplane in higher dimensions) that separates the two classes.

**For 2D data (two features):**

The decision boundary is where P(y=1|x) = 0.5, which occurs when:

```
w₀ + w₁x₁ + w₂x₂ = 0
```

Solving for x₂:

```
x₂ = -(w₀ + w₁x₁) / w₂
```

This is a straight line! Points on one side get classified as class 1, points on the other side as class 0.

**What the weights mean geometrically:**

- The vector [w₁, w₂] is perpendicular to the decision boundary
- It points toward the class 1 region
- The magnitude tells you how quickly probability changes as you move away from the boundary
- w₀ (bias) shifts the boundary parallel to itself

**Understanding through visualization:**

When I plotted the decision boundary for the first time on my spam detection data, everything clicked. I could see:

- Points far from the boundary: Model is very confident (probabilities near 0 or 1)
- Points near the boundary: Model is uncertain (probabilities near 0.5)
- Misclassified points: Usually near the boundary where classes overlap
- The effect of each feature: Moving in the direction of [w₁, w₂] increases P(y=1)

## Evaluation: Beyond Accuracy

My first logistic regression model had 95% accuracy! I was thrilled... until I realized my dataset was 95% class 0 and 5% class 1. A model that always predicted class 0 would also be 95% accurate, but completely useless for finding the rare class 1 examples (which were what I actually cared about!).

This taught me that accuracy alone is often meaningless for classification. I needed to understand:

**Precision:** Of the examples I predicted as class 1, how many actually were class 1?

- High precision = Few false alarms
- Important when false positives are costly
- Example: Drug safety testing (don't approve unsafe drugs)

**Recall:** Of all the actual class 1 examples, how many did I correctly identify?

- High recall = Caught most of the positives
- Important when false negatives are costly
- Example: Disease screening (don't miss sick patients)

**The precision-recall tradeoff:**

I can make precision perfect by only predicting class 1 when I'm absolutely certain (threshold = 0.99). But then recall suffers because I miss many class 1 examples that I wasn't quite confident enough about.

I can make recall perfect by predicting class 1 whenever I'm slightly leaning that way (threshold = 0.1). But then precision suffers because I get lots of false alarms.

The threshold (default 0.5) controls this tradeoff. Choosing the right threshold depends on your problem's costs of false positives versus false negatives.

**The confusion matrix:** This shows the full picture:

```
                    Predicted
                    0      1
Actual    0        TN     FP
          1        FN     TP
```

Every metric derives from these four numbers:

- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 Score = 2 × (Precision × Recall) / (Precision + Recall)

Understanding this matrix deeply was the key to evaluating my classifiers properly.

## 🔧 Files in This Folder

### `logistic_regression_scratch.py`

This contains the complete implementation using gradient descent. The key components are:

**The forward pass:**

1. Compute linear combination: z = Xw
2. Apply sigmoid: ŷ = σ(z)
3. Get probabilities for each class

**The backward pass (training):**

1. Compute loss: binary cross-entropy
2. Compute gradient: (1/n) X^T (ŷ - y)
3. Update weights: w = w - α × gradient
4. Repeat until convergence

**Key features:**

- Supports different gradient descent variants (batch, SGD, mini-batch)
- Tracks training history (loss, accuracy over time)
- Allows custom probability threshold for predictions
- Includes regularization support (L2 penalty to prevent overfitting)

**What I learned implementing this:**

The gradient is beautifully simple (just prediction error!), but getting the loss calculation right required careful attention to numerical stability. Computing log(0) causes problems, so I had to clip predictions to a small range like [1e-15, 1-1e-15].

I also discovered that feature scaling is even more important for logistic regression than linear regression. Because we're dealing with exponentials in the sigmoid, features with large magnitudes can cause numerical overflow. Standardizing features (mean=0, std=1) solved this completely.

### `logistic_regression_evaluation.py`

This file implements all the classification-specific evaluation metrics and analysis tools.

**Included metrics:**

- Confusion matrix (the foundation of everything)
- Accuracy, precision, recall, F1 score
- ROC curve and AUC (receiver operating characteristic)
- Precision-recall curve
- Classification report (comprehensive summary)

**ROC curve:** This plots the true positive rate against the false positive rate as you vary the threshold. The area under this curve (AUC) is a threshold-independent measure of model quality.

- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random guessing
- AUC < 0.5: Worse than random (you're predicting backwards!)

**When I use each metric:**

- Balanced classes: Accuracy works fine
- Imbalanced classes: Use precision, recall, F1
- Need single number: Use AUC (threshold-independent)
- Need to choose threshold: Look at precision-recall curve

### `decision_boundary_visualization.py`

This creates visual representations of how the model separates classes. Visualization was crucial for my understanding.

**What it can visualize:**

1. **2D decision boundaries**: Plot the line separating classes
2. **Probability contours**: Show regions of different confidence levels
3. **Decision regions**: Color code the space by predicted class
4. **Confidence visualization**: Darker colors = more confident predictions

**What these visualizations taught me:**

Linear decision boundaries work great when classes are linearly separable. But when classes overlap in complex ways, logistic regression struggles. The boundary is always a straight line (or hyperplane), so it can't capture complex curved decision regions.

This limitation led me to polynomial features (transform features to allow curved boundaries) and later to more complex models like decision trees and neural networks.

But for many real-world problems, linear boundaries work surprisingly well! Often classes are naturally separable, or close enough that the linear model performs excellently.

## 🎓 What I Learned Building This

### Initially, I Was Confused About...

**1. "Why is it called 'regression' if it does classification?"**

The name is historical. It's called "logistic regression" because we're regressing (modeling) the log-odds of the probability. The model predicts a probability (regression), which we then threshold to get a category (classification).

Mathematically: log(P/(1-P)) = w^T x

This is a linear relationship between features and log-odds. The inverse transformation (applying sigmoid) gives us probabilities.

**2. "Why can't I use linear regression and just round the output?"**

I tried this! It works poorly because:

- Linear regression outputs can be negative or greater than 1 (not valid probabilities)
- MSE loss isn't calibrated for classification
- The model is overly sensitive to outliers
- Probabilities aren't meaningful (can't interpret confidence levels)

Logistic regression fixes all these issues by design.

**3. "What's the difference between logistic regression and perceptron?"**

This confused me for ages! They're related but different:

Perceptron:

- Binary output: 0 or 1 (hard decision)
- Step activation: output is discrete
- No probability interpretation
- Historical algorithm from 1958

Logistic Regression:

- Probability output: between 0 and 1
- Sigmoid activation: smooth and differentiable
- Clear probability interpretation
- Much more flexible and useful

Logistic regression is strictly more powerful because you get both the probability and the decision.

**4. "How do I choose the decision threshold?"**

The default threshold is 0.5 (predict class 1 if P(y=1) ≥ 0.5). But this isn't always optimal!

If false positives are very costly (like approving a dangerous drug), use a higher threshold (maybe 0.8). This increases precision at the cost of recall.

If false negatives are very costly (like missing a cancer diagnosis), use a lower threshold (maybe 0.3). This increases recall at the cost of precision.

The ROC and precision-recall curves help you visualize this tradeoff and choose the threshold that's right for your problem.

### Things That Finally Clicked When I Implemented Them...

**1. The sigmoid is not arbitrary—it comes from probability theory**

The sigmoid function naturally emerges when you model the log-odds of probability as a linear function. If you assume:

```
log(P(y=1) / P(y=0)) = w^T x
```

And solve for P(y=1), you get exactly the sigmoid function! This isn't a random choice—it's the mathematically natural way to model binary probabilities with a linear model.

**2. Cross-entropy measures surprise from information theory**

Cross-entropy isn't just a loss function someone invented. It comes from information theory! It measures the expected "surprise" when predicting with your model's probabilities versus the true distribution.

When your predicted probability matches the true label, surprise is low. When they disagree, surprise is high. Cross-entropy is just the average surprise across all predictions.

This connection to information theory gave me a much deeper appreciation for why this loss function is the "correct" one for classification.

**3. The decision boundary is a hyperplane perpendicular to the weight vector**

Understanding the geometry was huge for me. The weight vector w points in the direction that most increases P(y=1). The decision boundary is the set of points where w^T x = 0, which is a hyperplane perpendicular to w.

This means:

- Moving in the direction of w increases the probability of class 1
- The magnitude of w controls how quickly probability changes
- The bias w₀ just shifts the boundary parallel to itself

Once I understood this geometry, I could intuitively understand what my model was learning by looking at the weights!

**4. Regularization is just as important as in regression**

With many features, logistic regression can overfit just like linear regression. The model learns to memorize the training data's quirks rather than the underlying pattern.

Adding L2 regularization (same as Ridge) penalizes large weights and forces the model to find simpler solutions that generalize better. The regularization strength controls the smoothness of the decision boundary.

## How to Use These Files

### Basic Binary Classification

```python
from logistic_regression_scratch import LogisticRegression
from foundations.data_preprocessing.feature_engineering import (
    Standardizer, add_bias_column, train_test_split
)

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize (critical for logistic regression!)
scaler = Standardizer()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add bias
X_train_final = add_bias_column(X_train_scaled)
X_test_final = add_bias_column(X_test_scaled)

# Train
model = LogisticRegression(learning_rate=0.1, n_epochs=500)
model.fit(X_train_final, y_train)

# Get probability predictions
probabilities = model.predict_proba(X_test_final)

# Get class predictions (using default threshold=0.5)
predictions = model.predict(X_test_final)

# Evaluate
from logistic_regression_evaluation import classification_report_binary
print(classification_report_binary(y_test, predictions))
```

### Tuning the Decision Threshold

```python
# Train model
model = LogisticRegression(learning_rate=0.1, n_epochs=500)
model.fit(X_train_final, y_train)

# Get probabilities
probabilities = model.predict_proba(X_test_final)

# Try different thresholds
for threshold in [0.3, 0.5, 0.7]:
    predictions = model.predict(X_test_final, threshold=threshold)

    precision = compute_precision(y_test, predictions)
    recall = compute_recall(y_test, predictions)

    print(f"Threshold {threshold}: Precision={precision:.3f}, Recall={recall:.3f}")

# Choose threshold based on your problem's cost structure
```

### Visualizing the Decision Boundary

```python
from decision_boundary_visualization import plot_decision_boundary

# For 2D data (2 features + bias)
plot_decision_boundary(
    model,
    X_test_final,
    y_test,
    title="Logistic Regression Decision Boundary"
)

# This shows:
# - The linear boundary separating classes
# - Probability contours (0.1, 0.3, 0.5, 0.7, 0.9)
# - Correctly and incorrectly classified points
# - Confidence regions (darker = more confident)
```

### Evaluating with ROC and AUC

```python
from logistic_regression_evaluation import compute_roc_curve, compute_auc

# Get probability predictions
probabilities = model.predict_proba(X_test_final)

# Compute ROC curve
fpr, tpr, thresholds = compute_roc_curve(y_test, probabilities)

# Compute AUC (area under ROC curve)
auc = compute_auc(fpr, tpr)

print(f"AUC: {auc:.4f}")
# Interpretation:
# AUC = 1.0: Perfect classifier
# AUC = 0.9: Excellent
# AUC = 0.8: Good
# AUC = 0.7: Fair
# AUC = 0.5: Random guessing
```

## 🔗 Dependencies on Foundations

This module builds on several foundation components:

**Already exist:**

- `Matrix`, `Vector` from `linear_algebra/vectors_and_matrices.py`
- `Standardizer`, `add_bias_column`, `train_test_split` from `data_preprocessing/`
- Gradient descent optimizers
- `sigmoid` from `activations.py`
- `binary_cross_entropy` from `loss_functions/cross_entropy_loss.py`
- Classification metrics from `metrics.py`

Everything needed is already in place from previous modules!

## Common Mistakes I Made (And Fixed)

### Mistake 1: Not Standardizing Features

My first attempt used raw features with very different scales (age in tens, income in thousands, etc.). The model failed to converge because the gradients were wildly different magnitudes.

Solution: Always standardize! Features should have mean=0 and std=1 before training.

### Mistake 2: Using Wrong Loss Function

I initially tried MSE because I was familiar with it from linear regression. Training was slow and the model got stuck.

Solution: Use binary cross-entropy! It's specifically designed for probability predictions and gives much better gradients.

### Mistake 3: Forgetting to Clip Probabilities

Computing log(0) or log(1) causes numerical errors (negative infinity). My training would crash when the model became very confident.

Solution: Clip predicted probabilities to [1e-15, 1-1e-15] before computing the loss. This prevents numerical overflow without affecting the model meaningfully.

### Mistake 4: Relying Only on Accuracy

My spam classifier had 95% accuracy! Turns out only 5% of emails were spam, so predicting "not spam" for everything also gave 95% accuracy. My model was useless.

Solution: Look at precision, recall, F1 score, and the confusion matrix. For imbalanced data, accuracy is often misleading.

### Mistake 5: Wrong Learning Rate

Too large (α > 1): Loss would oscillate or diverge wildly. The sigmoid can cause problems with large gradients.

Too small (α < 0.001): Training took forever, sometimes getting stuck in flat regions.

Solution: Start with α=0.1 for logistic regression. This is typically larger than for linear regression because the sigmoid naturally bounds the outputs.

## 💭 Key Insights

1. **Logistic regression is fundamentally about modeling probabilities**: The output is a probability between 0 and 1, which we can threshold to make a hard decision.

2. **The sigmoid transformation is the key innovation**: It converts unbounded linear outputs into valid probabilities, enabling probabilistic classification.

3. **Cross-entropy is the natural loss function**: It measures how well our probability predictions match the true labels, and gives clean gradients.

4. **The decision boundary is linear**: This is both a strength (simple, interpretable, fast) and a limitation (can't model complex curved boundaries without feature engineering).

5. **Evaluation is more subtle than regression**: Accuracy alone is often misleading. You need to understand precision, recall, F1, ROC, AUC, and choose the right metric for your problem.

6. **The threshold matters**: The default 0.5 threshold isn't always optimal. Adjust it based on the relative costs of false positives and false negatives.

7. **Feature scaling is critical**: More so than in linear regression, because exponentials are involved. Without standardization, training can be numerically unstable.

8. **Regularization prevents overfitting**: Just like linear regression, logistic regression can memorize training data. L2 regularization helps it generalize.

## 🎯 When to Use Logistic Regression

**Use logistic regression when:**

- You have binary classification (two classes)
- You need probability predictions (not just hard classifications)
- Classes are roughly linearly separable
- You want an interpretable model (can examine weights)
- You need fast training and prediction
- You have limited data (simple models generalize better)

**Consider alternatives when:**

- You need to model complex nonlinear boundaries (use polynomial features or more complex models)
- You have many classes (use softmax regression/multinomial logistic regression)
- Features have complex interactions (use decision trees or neural networks)
- You don't care about interpretability (ensemble methods might perform better)

---

Ready to dive into classification? Start with `logistic_regression_scratch.py` to see how it all works!
