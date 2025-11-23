"""
STOCHASTIC GRADIENT DESCENT (SGD): LEARNING FROM ONE SAMPLE AT A TIME
====================================================================

SGD revolutionized machine learning by making it possible to train on massive datasets.
Instead of using all data to compute a gradient (expensive!), we use just ONE random
sample per update (cheap!). This introduces noise but enables incredible scalability.

TEACHING PHILOSOPHY:
-------------------
We'll understand SGD from multiple perspectives:
1. COMPUTATIONAL: Why it's fast and scalable
2. STATISTICAL: How noise affects convergence
3. PRACTICAL: When SGD beats batch gradient descent
4. THEORETICAL: Variance-bias tradeoff in gradient estimation

THE FUNDAMENTAL DIFFERENCE:
--------------------------
**BATCH Gradient Descent:**
- Uses ALL n samples to compute gradient
- Gradient = (1/n) Σ ∇loss(θ, xᵢ, yᵢ)  [average over ALL data]
- Accurate gradient, slow per iteration
- Deterministic: same data → same gradient
- Requires O(n) computations per update

**STOCHASTIC Gradient Descent:**
- Uses ONE random sample to compute gradient
- Gradient ≈ ∇loss(θ, xᵢ, yᵢ)  [single random sample i]
- Noisy gradient, fast per iteration
- Stochastic: randomness in which sample is chosen
- Requires O(1) computations per update

For n = 1,000,000 samples:
- Batch GD: 1,000,000 computations per update
- SGD: 1 computation per update
- SGD is 1,000,000x faster per iteration!

WHY "STOCHASTIC"?
----------------
"Stochastic" means involving randomness. In SGD:
1. We randomly shuffle the dataset (or sample randomly)
2. Each update uses a random sample
3. The gradient estimate has variance due to this randomness

The gradient is now a **random variable**:
E[∇loss(θ, xᵢ, yᵢ)] = (1/n) Σ ∇loss(θ, xᵢ, yᵢ) = true gradient

In expectation, SGD gradient equals batch gradient, but individual updates are noisy!

THE NOISE IS ACTUALLY HELPFUL:
------------------------------
This seems bad (noisy gradients?!) but noise provides benefits:

**1. ESCAPES LOCAL MINIMA**
Batch GD deterministically goes downhill → stuck in local minimum
SGD bounces around due to noise → can escape local minima!

**2. ACTS AS REGULARIZATION**
Noise prevents overfitting by not letting model perfectly fit any one batch.
Similar to adding random perturbations during training.

**3. ENABLES ONLINE LEARNING**
Process data as it arrives, no need to store entire dataset.
Crucial for streaming data (stock prices, user activity).

**4. FASTER INITIAL PROGRESS**
Early in training, SGD makes rapid progress (many fast updates).
Near convergence, noise slows down (need learning rate decay).

CONVERGENCE ANALYSIS:
--------------------
Batch GD: Smooth descent to minimum
SGD: Noisy path that bounces around minimum

**Convergence rate:**
- Batch GD: O(1/k) after k iterations (linear convergence for convex)
- SGD: O(1/√k) after k iterations (slower due to variance)

But remember: SGD iterations are much faster!
- Batch GD: k slow iterations
- SGD: k fast iterations (can do more in same time)

**Convergence guarantee (convex case):**
With decaying learning rate αₖ = α₀ / (1 + k):
- SGD converges to optimum in expectation
- Variance decreases as learning rate decreases
- Never converges exactly (always bouncing due to noise)

THE LEARNING RATE PROBLEM:
-------------------------
Constant learning rate with SGD is problematic:

**Too large:**
- Makes big updates
- Bounces wildly around minimum
- Never converges precisely
- Can diverge if too large

**Too small:**
- Makes tiny updates
- Learns very slowly
- Gets stuck before reaching minimum
- Wastes computation

**Solution: Learning rate schedule (annealing)**
Start large (fast initial progress), decay over time (precise convergence).

Common schedules:
1. Step decay: α = α₀ * 0.5 every k epochs
2. Exponential: α = α₀ * e^(-kt)
3. 1/t decay: α = α₀ / (1 + kt)
4. Polynomial: α = α₀ / (1 + t)^p

EPOCHS AND ITERATIONS:
---------------------
**Epoch**: One complete pass through entire dataset
**Iteration**: One parameter update

For n samples:
- Batch GD: 1 iteration = 1 epoch (uses all data)
- SGD: n iterations = 1 epoch (n single-sample updates)

Typical training: 100s of epochs for SGD, 10s for batch GD.

SGD VS MINI-BATCH GD:
--------------------
Pure SGD (batch size = 1) is rarely used in practice!
Most "SGD" implementations actually use mini-batches:

**Mini-batch GD:** Use b samples per update (e.g., b = 32)
- b = 1: Pure SGD (too noisy)
- b = n: Batch GD (too slow)
- b = 32-256: Sweet spot (practical "SGD")

Mini-batching provides:
- Less variance than SGD
- Vectorization speedup (GPU parallel computation)
- Still scalable to large datasets

VARIANCE VS BIAS IN GRADIENT:
----------------------------
Gradient estimate has bias and variance:

**Batch gradient:**
- Bias: 0 (exact gradient of empirical risk)
- Variance: 0 (deterministic)
- Problem: Expensive to compute

**SGD gradient:**
- Bias: 0 (unbiased in expectation)
- Variance: σ²/1 (high variance from single sample)
- Problem: Noisy updates

**Mini-batch gradient (size b):**
- Bias: 0 (still unbiased)
- Variance: σ²/b (reduced by factor of b)
- Sweet spot: Some noise (good) but not too much

WHEN TO USE SGD:
---------------

✓ **Large datasets** (n > 10,000)
  Batch GD too slow, SGD makes training feasible

✓ **Online learning** (streaming data)
  Update model as new data arrives

✓ **Non-convex optimization** (neural networks)
  Noise helps escape local minima

✓ **Memory constraints**
  Can't load entire dataset into memory

✗ **Small datasets** (n < 1,000)
  Batch GD works fine, less noise is better

✗ **Need exact gradient**
  Scientific computing, sensitive optimization

✗ **Convex with Hessian info available**
  Second-order methods (Newton, L-BFGS) are better

PRACTICAL TIPS:
--------------

1. **Shuffle data each epoch**
   Prevents learning order patterns

2. **Start with larger learning rate**
   ~0.1 for normalized data, ~0.01 for raw data

3. **Use learning rate schedule**
   Reduce every 10-30 epochs by factor of 2-10

4. **Monitor training loss**
   Should decrease on average (but will be noisy)

5. **Use validation set**
   Check generalization (validation loss should decrease)

6. **Consider mini-batches**
   Batch size 32-256 often better than pure SGD

7. **Use momentum or Adam**
   Modern optimizers reduce SGD variance

Let's implement SGD with learning rate schedules!
"""
