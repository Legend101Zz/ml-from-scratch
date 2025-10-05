Okay day3 ( because foundations took 2 days) and now we are into one of the most simple yet elegant classical ML algos KNN

We’ll move in three phases:

1. The _intuition_ of what KNN really means.
2. The _mathematical and visual sense_ of it.
3. The _practical example_ that shows it in action.

---

## 🧠 1. The Deep Intuition — What Is Learning by “Neighbors”?

---

### 📍Scenario: A New Person in Town

Suppose you’ve just moved into a new city.  
You don’t know anyone — but you see a few people around and notice their habits.

You notice some people:

- go to the arcade a lot,
- play video games,
- rarely go hiking.

Others:

- go running every morning,
- spend weekends outdoors,
- rarely play games.

Now a **new person** moves into your apartment building.  
You don’t know if they’re a “gamer” or an “outdoorsy” person.

But you do know **how much time they spend playing games** and **how much time they spend outdoors**.

---

### 💡What would you do?

You’d think:

> “Hmm, this person spends 8 hours gaming a week and 2 hours outdoors.  
> I know a few people like that — and they were mostly gamers.  
> So I’ll guess this person is a gamer too.”

That’s the essence of **k-Nearest Neighbors**.

It’s a _learning method without learning rules_.

---

### ⚙️ It’s Not a Formula — It’s a Comparison

Unlike algorithms that build an equation (like linear regression or SVM),  
k-NN **doesn’t learn a function** during training.

It just **remembers all the data points**.

When a new data point comes along, it compares it to all the known ones,  
finds the _k_ most similar examples, and predicts the majority class.

That’s why we call it **lazy learning** —  
it postpones “learning” until it needs to make a prediction.

---

## 🧭 2. The Geometry of “Neighbors”

Let’s visualize what’s really happening.

Imagine you have this dataset:

| Person | Game hours | Outdoor hours | Label    |
| ------ | ---------- | ------------- | -------- |
| A      | 9          | 1             | Gamer    |
| B      | 8          | 2             | Gamer    |
| C      | 1          | 10            | Outdoors |
| D      | 2          | 9             | Outdoors |

Now you meet **E**, who plays 7 hours of games and spends 3 hours outdoors.  
You want to predict E’s label.

---

### Step 1: Plot It Mentally

If we plot “game hours” on the x-axis and “outdoor hours” on the y-axis,  
then:

- Gamers cluster on the **bottom-right** (lots of games, few outdoor hours),
- Outdoorsy people cluster on the **top-left** (few games, many outdoor hours).

E would land near the “gamer” group.

So, intuitively, **E is probably a gamer**.

---

### Step 2: Measure Closeness (Distance)

We formalize “similarity” as **distance** between points.

For E(7, 3):

- Distance to A(9,1)  
   = √((9−7)² + (1−3)²) = √(4 + 4) = √8 ≈ 2.83
- Distance to B(8,2)  
   = √((8−7)² + (2−3)²) = √(1 + 1) = √2 ≈ 1.41
- Distance to C(1,10)  
   = √((1−7)² + (10−3)²) = √(36 + 49) = √85 ≈ 9.22
- Distance to D(2,9)  
   = √((2−7)² + (9−3)²) = √(25 + 36) = √61 ≈ 7.81

The smallest distances are to **B and A**, both labeled _Gamer_.

---

### Step 3: Voting

Let’s say _k = 3_ (we take the 3 nearest neighbors).

The three closest are:

1. B (Gamer)
2. A (Gamer)
3. D (Outdoors)

So, two gamers and one outdoorsy → the **majority vote is “Gamer.”**

---

### Step 4: Why It Works

It works because of a simple assumption:

> “Similar inputs tend to have similar outputs.”

If two people have similar habits, they likely have similar preferences.

This assumption — called the **continuity assumption** — is what all nonparametric algorithms rely on.

---

## 🧮 3. Mathematical Intuition

You can think of KNN as defining **regions** in the feature space.

For every new point, we look at which _region_ (neighborhood) it falls into.

When you set (k = 1), the boundary between classes is very sharp — each training point “owns” the space closest to it.

When you increase (k), the boundary becomes smoother and more robust to noise.

That’s why choosing (k) is about balancing **bias and variance**:

- Small (k): low bias, high variance (sensitive to noise)
- Large (k): high bias, low variance (over-smoothing)

---

## 💻 4. Translating Intuition into Code

Let’s now connect intuition to a simple program.

Here’s the dataset we imagined:

```python
from numpy import array
import operator

def createDataSet():
    group = array([
        [9, 1],
        [8, 2],
        [1, 10],
        [2, 9]
    ])
    labels = ['Gamer', 'Gamer', 'Outdoors', 'Outdoors']
    return group, labels
```

And here’s our classifier:

```python
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = inX - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]
```

Now classify our new friend E(7, 3):

```python
group, labels = createDataSet()
print(classify0(array([7, 3]), group, labels, 3))
```

Output:

```
Gamer
```

Just like our intuition.

---

## 🔍 5. Understanding Why KNN Is “Lazy”

Notice that KNN didn’t **train** anything.  
There’s no equation or model to store. It just keeps the dataset in memory.

When asked to classify a new point, it:

1. Computes distances to all known points,
2. Finds the closest ones,
3. Votes.

That’s why it’s sometimes called a **memory-based** or **instance-based** learner.

---

## ⚖️ 6. Pros and Cons

**Advantages:**

- Simple, intuitive, no training phase.
- Works well when the decision boundary is irregular.
- Naturally handles multi-class problems.

**Disadvantages:**

- Slow for large datasets (needs to compare to every point).
- Sensitive to the scale of features (so normalization is vital).
- Doesn’t handle irrelevant features well — each one affects distance.

---

Now let's code this

## Data Normalization: Making Features Comparable

One critical concept for k-NN is that all features need to be on comparable scales. Imagine predicting house prices using both square footage (1000-5000) and number of bedrooms (1-5). The square footage would dominate the distance calculation simply because its values are larger, not because it's more important.
go to preprocessing.py:
