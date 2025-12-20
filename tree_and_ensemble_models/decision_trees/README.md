# Decision Trees: Learning by Asking Questions

> "The art and science of asking questions is the source of all knowledge." — Thomas Berger

---

## What Are Decision Trees?

Imagine you're trying to predict whether someone will buy a product. Instead of fitting a mathematical equation, you ask a series of yes/no questions:

```
Is their income > $50k?
├─ Yes: Are they interested in tech?
│  ├─ Yes: PREDICT: Will buy (confidence: 85%)
│  └─ No: Is their age < 30?
│     ├─ Yes: PREDICT: Will buy (confidence: 60%)
│     └─ No: PREDICT: Won't buy (confidence: 70%)
└─ No: PREDICT: Won't buy (confidence: 90%)
```

That's a decision tree! It learns to ask the right questions in the right order to make predictions.

---

## Why Decision Trees Are Beautiful

### 1. **Human-Interpretable**

Unlike neural networks (black boxes) or linear models (opaque equations), you can literally read a decision tree and understand exactly why it made a prediction.

When I showed my first decision tree to a non-technical colleague, they immediately understood it. "Oh, so it's like a flowchart!" Exactly!

### 2. **No Feature Scaling Required**

Linear models need features on similar scales. Neural networks need normalization. Trees? They don't care!

Why? Because trees only compare values within a single feature:

- "Is feature[2] > 5.5?" works whether feature[2] is in [0, 1] or [0, 1000000]
- The tree just cares about relative ordering, not absolute magnitude

### 3. **Handles Non-Linear Relationships Naturally**

Linear models draw straight lines. Trees draw rectangles!

Example: XOR problem (linear models fail, trees solve easily)

```
X1  X2  Label
0   0   0
0   1   1
1   0   1
1   1   0
```

A tree solves this with just 2 splits:

```
Is X1 > 0.5?
├─ No: Predict X2 (if X2 > 0.5 → 1, else → 0)
└─ Yes: Predict NOT X2 (if X2 > 0.5 → 0, else → 1)
```

### 4. **Automatic Feature Interaction**

Trees naturally capture interactions. If the rule is:

```
Income > 50k AND Age < 30 → High risk
```

The tree learned this interaction automatically - no need for manual feature engineering!

---

## The Core Algorithm: Recursive Binary Splitting

Building a decision tree is surprisingly simple:

```
function BuildTree(data):
    if stopping_criterion_met:
        return LeafNode(predict majority class or mean value)

    best_split = find_best_split(data)

    left_data = data where feature[best_split.feature] <= best_split.threshold
    right_data = data where feature[best_split.feature] > best_split.threshold

    left_child = BuildTree(left_data)
    right_child = BuildTree(right_data)

    return InternalNode(best_split, left_child, right_child)
```

The magic is in `find_best_split()` - how do we decide which feature and threshold to split on?

---

## The Split Criterion: Measuring Impurity

We want splits that create "pure" groups - where samples in each group are similar.

### For Classification: Gini Impurity

**Formula:**

```
Gini = 1 - Σ(p_i)²
```

Where p_i is the proportion of class i in the node.

**Intuition:** "What's the probability of misclassifying a random sample if we label it randomly according to the class distribution?"

**Examples:**

Pure node (all samples same class):

```
[100 samples of class A, 0 samples of class B]
p_A = 1.0, p_B = 0.0
Gini = 1 - (1.0² + 0.0²) = 1 - 1 = 0.0  ← Perfect!
```

Completely impure (50/50 split):

```
[50 samples of class A, 50 samples of class B]
p_A = 0.5, p_B = 0.5
Gini = 1 - (0.5² + 0.5²) = 1 - 0.5 = 0.5  ← Worst!
```

**Why squared?** It penalizes mixed distributions more heavily than linear would.

### For Classification: Entropy

**Formula:**

```
Entropy = -Σ p_i × log₂(p_i)
```

**Intuition:** "How much information do I gain by learning which class this sample belongs to?"

**Information Theory Connection:**

- Entropy measures uncertainty
- Pure nodes have 0 uncertainty (Entropy = 0)
- Mixed nodes have high uncertainty (Entropy = 1 for 50/50 binary)

**Examples:**

Pure node:

```
[100 samples of class A]
p_A = 1.0
Entropy = -(1.0 × log₂(1.0)) = 0.0  ← No uncertainty!
```

50/50 split:

```
[50 A, 50 B]
p_A = 0.5, p_B = 0.5
Entropy = -(0.5 × log₂(0.5) + 0.5 × log₂(0.5))
        = -(0.5 × (-1) + 0.5 × (-1))
        = -(-0.5 - 0.5) = 1.0  ← Maximum uncertainty!
```

### Gini vs. Entropy: Which to Use?

In practice, they give very similar results! The differences:

**Gini:**

- Faster to compute (no logarithms)
- Range: [0, 0.5] for binary classification
- Default in scikit-learn

**Entropy:**

- Slightly better at finding pure splits
- Range: [0, 1] for binary classification
- More theoretically grounded (information theory)

I usually use Gini because it's faster and the difference is negligible.

### For Regression: Mean Squared Error (MSE)

**Formula:**

```
MSE = (1/n) × Σ(y_i - ȳ)²
```

Where ȳ is the mean of targets in the node.

**Intuition:** "How much do the targets vary around their mean?"

**Example:**

Pure node (all samples have same target):

```
Targets: [5.0, 5.0, 5.0]
Mean: 5.0
MSE = (1/3) × [(5-5)² + (5-5)² + (5-5)²] = 0.0  ← Perfect!
```

Mixed node:

```
Targets: [1.0, 5.0, 9.0]
Mean: 5.0
MSE = (1/3) × [(1-5)² + (5-5)² + (9-5)²]
    = (1/3) × [16 + 0 + 16] = 10.67  ← High variance!
```

---

## Information Gain: Evaluating Splits

Once we compute impurity, how do we choose the best split?

**Information Gain = Parent Impurity - Weighted Average of Child Impurities**

```
Gain = Impurity(parent) - [
    (n_left/n_total) × Impurity(left) +
    (n_right/n_total) × Impurity(right)
]
```

We want to **maximize** information gain - find splits that reduce impurity the most!

**Example:**

Parent node: 100 samples [60 class A, 40 class B]

```
Gini(parent) = 1 - (0.6² + 0.4²) = 1 - 0.52 = 0.48
```

Split 1: Feature X > 5

- Left: 70 samples [50 A, 20 B] → Gini = 1 - (0.71² + 0.29²) = 0.41
- Right: 30 samples [10 A, 20 B] → Gini = 1 - (0.33² + 0.67²) = 0.44

```
Gain = 0.48 - [0.7×0.41 + 0.3×0.44]
     = 0.48 - [0.287 + 0.132]
     = 0.48 - 0.419 = 0.061
```

Split 2: Feature Y > 3

- Left: 50 samples [45 A, 5 B] → Gini = 1 - (0.9² + 0.1²) = 0.18
- Right: 50 samples [15 A, 35 B] → Gini = 1 - (0.3² + 0.7²) = 0.42

```
Gain = 0.48 - [0.5×0.18 + 0.5×0.42]
     = 0.48 - 0.30 = 0.18  ← Much better!
```

We choose Split 2 because it has higher information gain!

---

## Stopping Criteria: When to Stop Splitting

Without stopping criteria, trees grow until every leaf has exactly one sample - perfect training accuracy, terrible generalization (overfitting)!

We stop splitting when ANY of these are true:

### 1. **Maximum Depth Reached**

```python
if current_depth >= max_depth:
    return LeafNode()
```

**Why?** Limits tree complexity. Deeper trees = more complex = higher variance.

**Typical values:** 5-20 (I usually start with 10)

### 2. **Minimum Samples per Node**

```python
if n_samples < min_samples_split:
    return LeafNode()
```

**Why?** Small nodes might just be noise. Don't split them!

**Typical values:** 2-20 (I usually use 5)

### 3. **Pure Node (Classification)**

```python
if all_samples_same_class:
    return LeafNode()
```

**Why?** Can't improve a perfect node!

### 4. **No Information Gain**

```python
if max_gain <= 0:
    return LeafNode()
```

**Why?** If no split reduces impurity, don't split!

### 5. **Minimum Samples per Leaf**

```python
if n_left < min_samples_leaf or n_right < min_samples_leaf:
    don't_use_this_split
```

**Why?** Leaves with 1-2 samples are probably overfitting.

---

## The Overfitting Problem

Decision trees have incredibly high capacity - they can memorize any dataset!

**Example:**

Training data (with noise):

```
X = [1, 2, 3, 4, 5]
y = [2, 4, 5, 8, 10]  ← Target = 2×X, but y[2]=5 is noise (should be 6)
```

A deep tree will create a rule specifically for X=3:

```
if X == 3:
    predict 5  ← Memorized the noise!
```

**Result:**

- Training accuracy: 100%
- Test accuracy: 60%

**Solution:** Limit tree depth, use ensemble methods (Random Forest, Gradient Boosting)

---

## Classification vs. Regression Trees

The algorithm is nearly identical, only the impurity measure and leaf predictions differ:

| Aspect              | Classification               | Regression       |
| ------------------- | ---------------------------- | ---------------- |
| **Impurity**        | Gini or Entropy              | MSE or MAE       |
| **Leaf prediction** | Majority class               | Mean of targets  |
| **Output**          | Class label or probabilities | Continuous value |

---

## My Implementation Strategy

I implemented trees in three stages:

### Stage 1: Core Splitting Logic

- Find best split for a given dataset
- Compute impurity measures (Gini, Entropy, MSE)
- Handle edge cases (pure nodes, no valid splits)

### Stage 2: Recursive Tree Building

- Build tree recursively
- Implement stopping criteria
- Handle both classification and regression

### Stage 3: Prediction and Visualization

- Traverse tree to make predictions
- Print tree structure for debugging
- Add tree statistics (depth, number of leaves)

---

## Key Insights

### 1. **Greedy Algorithm Limitations**

Trees use **greedy splitting** - at each node, pick the split that reduces impurity most RIGHT NOW.

This is not optimal! Sometimes a split that looks bad now enables a great split later.

**Example:**

```
XOR problem requires 2 splits to solve.
First split might have 0 information gain!
A greedy algorithm might stop.
```

But in practice, greedy works surprisingly well!

### 2. **Axis-Aligned Splits Only**

Trees only split parallel to feature axes: `X[i] <= threshold`

They can't learn diagonal boundaries naturally:

```
Class A: samples where X₁ + X₂ > 5
Class B: samples where X₁ + X₂ <= 5
```

A tree would need many splits to approximate this diagonal line.

**Solution:** Feature engineering (add X₁ + X₂ as a feature) or use ensemble methods.

### 3. **Trees Love Categorical Features**

For categorical features with K categories, trees can split K ways efficiently.

Linear models need K-1 one-hot encoded features (curse of dimensionality).

Trees just split: "Is category in {A, B, C}?"

### 4. **Feature Importance for Free**

While building the tree, we track how much each feature reduced impurity.

**Feature importance = Sum of impurity reductions from all splits using that feature**

This gives us automatic feature ranking!

---

## When to Use Decision Trees

**Use trees when:**

- Interpretability matters (need to explain predictions)
- Features have complex interactions
- No time for feature scaling/engineering
- Mix of numerical and categorical features

**Avoid single trees when:**

- Need maximum accuracy (use Random Forest or XGBoost instead)
- Data is high-dimensional and sparse (trees don't work well)
- Want probabilistic predictions (tree probabilities are coarse)

---

## What's Next

After implementing decision trees, we'll build:

1. **Ensemble Basics** - Combining multiple trees
2. **Random Forest** - Parallel trees with decorrelation
3. **Boosting** - Sequential trees learning from mistakes
4. **XGBoost** - State-of-the-art gradient boosting

Each builds on the decision tree foundation!

---
