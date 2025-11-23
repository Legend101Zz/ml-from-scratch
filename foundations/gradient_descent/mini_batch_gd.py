"""
MINI-BATCH GRADIENT DESCENT: THE GOLDILOCKS OPTIMIZER
=====================================================

This module implements Mini-Batch Gradient Descent (MBGD), which is the most widely
used optimization algorithm in modern machine learning and deep learning.

THE CORE PHILOSOPHY:
-------------------
"Neither too hot, nor too cold, but just right."

Mini-Batch GD is the perfect compromise between two extremes:
- Batch GD: Too slow (processes ALL data before one update)
- SGD: Too noisy (processes ONE sample, very chaotic updates)

Mini-Batch GD: Just right! (processes a SMALL GROUP of samples, balanced updates)

THE GOLDILOCKS ANALOGY:
----------------------
Imagine you're a teacher grading student essays to improve your grading rubric:

**Batch GD (Papa Bear's porridge - too hot):**
"I'll read all 1,000 essays, calculate the average quality across everything,
then update my rubric once."
→ Extremely accurate feedback, but painfully slow!

**SGD (Mama Bear's porridge - too cold):**
"I'll read one random essay, immediately update my rubric, repeat."
→ Super fast feedback, but each essay gives wildly different signals!

**Mini-Batch GD (Baby Bear's porridge - just right!):**
"I'll read 32 essays at a time, calculate the average quality of those 32,
then update my rubric. Repeat with the next 32."
→ Fast enough (only 32 essays per update), stable enough (averaging reduces noise)!

WHY THIS IS THE INDUSTRY STANDARD:
----------------------------------
If you look at ANY modern deep learning framework (PyTorch, TensorFlow, JAX), 
they ALL default to mini-batch gradient descent. Here's why:

1. **Hardware Efficiency (GPUs love mini-batches!)**
   Modern GPUs are designed to process matrices in parallel. When you give a GPU
   32 samples at once, it processes them simultaneously! With SGD (one sample),
   you're wasting 99% of the GPU's parallel processing power.
   
   Think of it like this: A GPU is like a bus with 32 seats. 
   - SGD: Only one person rides the bus (wasteful!)
   - Mini-Batch (32): Bus is full (efficient!)
   - Batch GD (10,000): Trying to fit 10,000 people on a 32-seat bus (impossible!)

2. **Noise Reduction (Smoother than SGD)**
   By averaging gradients over a batch, we reduce variance. If 3 samples point
   slightly wrong but 29 samples point correctly, the average points correctly!
   
   Mathematical intuition: The variance of an average decreases as 1/√n.
   So a batch of 32 has variance that's √32 ≈ 5.6× smaller than SGD!

3. **Memory Efficiency (Better than Batch GD)**
   We don't need to load ALL data into memory at once. We process manageable chunks.
   With a million samples, Batch GD might overflow memory. Mini-Batch (32 at a time)
   fits comfortably.

4. **Convergence Speed (Best of both worlds)**
   - Faster than Batch GD: More updates per epoch (n/batch_size updates instead of 1)
   - Smoother than SGD: Less oscillation, more stable convergence
   - Just right: Gets to "good enough" quickly AND settles near the optimum

THE MATHEMATICAL INTUITION:
--------------------------
In Batch GD, the gradient is exact:
    ∇J(w) = (1/n) Σᵢ₌₁ⁿ ∇Jᵢ(w)
    
In SGD, the gradient is a noisy sample:
    ∇J(w) ≈ ∇Jᵢ(w)  (high variance, unbiased)
    
In Mini-Batch GD, the gradient is a mini-batch average:
    ∇J(w) ≈ (1/m) Σᵢ∈batch ∇Jᵢ(w)  where m = batch_size
    
This mini-batch average has:
- LOWER variance than SGD (we're averaging m samples, not 1)
- HIGHER variance than Batch GD (we're using m samples, not all n)
- FASTER updates than Batch GD (we update n/m times per epoch, not once)

THE BATCH SIZE HYPERPARAMETER:
-----------------------------
Choosing batch size is an art informed by theory:

**Powers of 2 (32, 64, 128, 256):**
Computer memory is organized in powers of 2, so these sizes are most efficient
for GPU/CPU hardware. Always prefer 32 over 30, or 64 over 50!

**Small batches (8-32):**
+ More updates per epoch → faster convergence
+ Higher noise → better exploration, might escape local minima
+ Less memory needed
- Harder to parallelize efficiently
- Noisier convergence

**Medium batches (32-128):**
+ Good balance of speed and stability
+ Efficient GPU utilization
+ Standard choice for most problems
This is the sweet spot for most applications!

**Large batches (256-1024):**
+ Very smooth gradients → stable convergence
+ Maximum GPU efficiency
- Fewer updates per epoch → slower learning
- Might get stuck in sharp minima (poor generalization)
- Need to increase learning rate proportionally

**Rule of thumb:** Start with 32. If training is noisy, increase to 64 or 128.
If training is too slow, try 16. If you have huge GPUs, try 256.

THE SHUFFLE-AND-BATCH DANCE:
---------------------------
One of the most important (and often overlooked) details in mini-batch GD is
how we create batches each epoch:

**CORRECT APPROACH (what we implement):**
1. Shuffle ALL indices at the start of each epoch
2. Divide shuffled indices into sequential batches
3. Process batches in order

This ensures:
- Each sample appears exactly once per epoch
- Samples appear in different batches each epoch
- No systematic bias from batch composition

**WRONG APPROACH (common mistake):**
1. Create fixed batches at the start
2. Shuffle batch order each epoch

This means the same samples always appear together, which can introduce bias!
For example, if all "cat" images ended up in batch 1 by chance, the model would
see that skewed distribution every epoch.

HANDLING THE LAST INCOMPLETE BATCH:
-----------------------------------
When n_samples doesn't divide evenly by batch_size, the last batch is smaller.

Example: 100 samples, batch_size=32
- Batch 1: samples 0-31 (32 samples)
- Batch 2: samples 32-63 (32 samples)  
- Batch 3: samples 64-95 (32 samples)
- Batch 4: samples 96-99 (only 4 samples!)

We have two choices:

**Option 1: Use the incomplete batch (our approach)**
Process all samples, even if the last batch is small. This ensures every sample
contributes to training every epoch.

**Option 2: Drop the last batch**
Ignore samples that don't fit evenly. Simpler, but you lose data!

Option 1 is better for small datasets. Option 2 is fine for huge datasets where
losing a few samples per epoch doesn't matter.

WHEN TO USE MINI-BATCH GD:
-------------------------
✅ Almost always! This is the default choice for modern machine learning.
✅ Any dataset larger than a few hundred samples
✅ When training neural networks (leverages GPU parallelization)
✅ When you want both speed AND stability
✅ When memory is limited (can't load all data at once)

❌ Very tiny datasets (< 100 samples): Batch GD is fine, simpler
❌ Streaming/online learning: SGD is more appropriate
❌ When you specifically WANT noise (e.g., escaping sharp minima): Use smaller batches

Author: Mrigesh
Philosophy: Balance is the key to practical machine learning
"""

import math
import random

from foundations.linear_algebra.vectors_and_matrices import Matrix, Vector

from .loss_strategies import LossFunction


class MiniBatchGradientDescent:
    """
    Mini-Batch Gradient Descent: The pragmatic optimizer that balances speed and stability.
    
    CORE ALGORITHM:
    --------------
    For each epoch:
        1. Shuffle all samples randomly
        2. Divide samples into mini-batches of size m
        3. For each mini-batch:
            a. Compute predictions for the batch (m samples)
            b. Compute average gradient over the batch
            c. Update weights once per batch
    
    THE KEY DIFFERENCES:
    -------------------
    Batch GD:      n samples → 1 gradient → 1 update per epoch
    SGD:           n samples → n gradients → n updates per epoch (one per sample)
    Mini-Batch GD: n samples → n/m gradients → n/m updates per epoch
    
    Example with 1000 samples:
    - Batch GD: 1 update per epoch
    - SGD: 1000 updates per epoch
    - Mini-Batch (m=32): 31 updates per epoch (1000/32 ≈ 31)
    
    Mini-Batch does 31× more updates than Batch GD (faster learning)
    But only 1/32 the updates of SGD (more stable)
    
    THE VECTORIZATION ADVANTAGE:
    ---------------------------
    Unlike SGD which processes one sample at a time, Mini-Batch processes multiple
    samples simultaneously. This is crucial for performance:
    
    **Sequential processing (SGD):**
    For each sample:
        prediction = compute(sample)  ← can't parallelize
        
    **Parallel processing (Mini-Batch):**
    predictions = compute(all_samples_in_batch)  ← GPU processes all at once!
    
    On a GPU, computing 32 samples simultaneously takes almost the same time as
    computing 1 sample! This is why Mini-Batch is so much faster than SGD in practice,
    even though they do similar numbers of updates.
    
    WHY BATCH SIZE MATTERS:
    ----------------------
    Batch size is a critical hyperparameter that affects:
    
    1. **Convergence Speed:**
       - Small batches (16-32): Fast convergence, more updates per epoch
       - Large batches (256-512): Slower convergence, fewer updates per epoch
    
    2. **Gradient Quality:**
       - Small batches: Noisier gradients (like SGD)
       - Large batches: Smoother gradients (like Batch GD)
    
    3. **Generalization:**
       - Small batches: More noise → better exploration → better generalization
       - Large batches: Less noise → might overfit to sharp minima
    
    4. **Hardware Efficiency:**
       - Powers of 2 (32, 64, 128, 256) align with GPU memory architecture
       - Non-powers of 2 waste memory and computation
    
    5. **Memory Usage:**
       - Batch size = 32: Can train on almost any GPU
       - Batch size = 1024: Needs powerful GPU with lots of memory
    
    LEARNING RATE SCALING:
    ---------------------
    An advanced topic: when you change batch size, you might need to adjust learning rate!
    
    Intuition: Larger batches give more accurate gradients, so you can take bigger steps.
    Rule of thumb: If you double batch size, you can increase learning rate by √2.
    
    Example:
    - Batch size 32, learning rate 0.01 → good
    - Batch size 128 (4× larger), learning rate 0.02 (2× larger, since √4=2) → also good
    
    PARAMETERS:
    ----------
    loss_function : LossFunction
        The strategy for computing loss and gradients (MSE, MAE, Huber, etc.)
        
    batch_size : int (default: 32)
        Number of samples to process together before updating weights.
        
        Recommended values:
        - 8-16: Small datasets (< 1000 samples), want more updates
        - 32-64: Standard choice for most problems
        - 128-256: Large datasets, powerful GPUs
        - 512-1024: Huge datasets, multiple GPUs, advanced users only
        
        Always prefer powers of 2 for hardware efficiency!
        
    learning_rate : float (default: 0.01)
        Step size for each update.
        
        Mini-Batch typically uses learning rates between SGD and Batch GD:
        - Too small (0.0001): Slow convergence
        - Good (0.01-0.1): Steady progress
        - Too large (1.0): Oscillation or divergence
        
    n_epochs : int (default: 100)
        Number of complete passes through the dataset.
        Each epoch processes all n samples in batches of size m.
        
    verbose : bool (default: False)
        If True, print loss periodically during training.
        
    ATTRIBUTES (learned during training):
    ------------------------------------
    weights_ : Vector
        The learned parameter vector after training.
        
    history_ : dict
        Training history containing:
        - 'loss': Loss values (computed on full dataset periodically)
        - 'epoch': Epoch numbers
        
        Note: Computing loss on full dataset is expensive, so we only do it
        occasionally (every few epochs) for monitoring, not after every batch.
    """
    
    def __init__(
        self, 
        loss_function: LossFunction, 
        batch_size: int = 32, 
        learning_rate: float = 0.01, 
        n_epochs: int = 100, 
        verbose: bool = False
    ):
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.verbose = verbose
        
        # These will be set during training
        self.weights_ = None
        self.history_ = {'loss': [], 'epoch': []}
        
        # Validate batch size
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        
        # Warning for non-power-of-2 batch sizes
        # Powers of 2: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, ...
        # These align with computer memory architecture for efficiency
        is_power_of_2 = (batch_size & (batch_size - 1)) == 0 and batch_size > 0
        if not is_power_of_2 and batch_size > 4:
            import warnings
            warnings.warn(
                f"batch_size={batch_size} is not a power of 2. "
                f"Consider using 32, 64, 128, or 256 for better hardware efficiency."
            )
    
    def fit(self, X: Matrix, y: Matrix) -> 'MiniBatchGradientDescent':
        """
        Fit the model using Mini-Batch Gradient Descent.
        
        This is where the balanced learning happens! We process data in small groups
        (mini-batches), averaging their gradients to get stable-yet-frequent updates.
        
        PARAMETERS:
        ----------
        X : Matrix, shape (n_samples, n_features)
            Training data where each row is a sample, each column is a feature.
            
        y : Matrix, shape (n_samples, 1)
            Target values (labels) for each sample.
            
        RETURNS:
        -------
        self : MiniBatchGradientDescent
            Returns self for method chaining (sgd.fit(X,y).predict(X_test))
        
        THE ALGORITHM IN DETAIL:
        -----------------------
        For each epoch:
            1. Shuffle all n samples to break any ordering patterns
            2. Divide into batches: [batch_1, batch_2, ..., batch_k]
               where k = ceil(n / batch_size)
            3. For each batch (containing m samples):
                a. Compute predictions for all m samples in batch
                b. Compute errors for all m samples
                c. Compute average gradient over these m samples
                d. Update weights using this batch-averaged gradient
            4. After all batches, optionally compute full loss for monitoring
        
        THE SHUFFLE-BATCH-UPDATE CYCLE:
        ------------------------------
        Epoch 1:
          Shuffle: [s3, s1, s5, s2, s4, s6, ...]
          Batch 1: [s3, s1] → update
          Batch 2: [s5, s2] → update
          Batch 3: [s4, s6] → update
          ...
          
        Epoch 2:
          Shuffle: [s2, s6, s1, s4, s3, s5, ...] (different order!)
          Batch 1: [s2, s6] → update (different samples than epoch 1's batch 1)
          Batch 2: [s1, s4] → update
          Batch 3: [s3, s5] → update
          ...
        
        This ensures samples appear in different contexts each epoch!
        
        Example:
        -------
        >>> X = Matrix([[1, 2], [1, 3], [1, 4], [1, 5]])  # 4 samples
        >>> y = Matrix([[5], [7], [9], [11]])  # y = 1 + 2x
        >>> mbgd = MiniBatchGradientDescent(
        ...     loss_function=MSELoss(),
        ...     batch_size=2,     # Process 2 samples at a time
        ...     learning_rate=0.01,
        ...     n_epochs=100
        ... )
        >>> mbgd.fit(X, y)
        >>> # With 4 samples and batch_size=2, we make 2 updates per epoch
        >>> # Total updates = 100 epochs × 2 updates/epoch = 200 updates
        """
        
        # =========================================================================
        # STEP 0: INITIALIZATION AND SETUP
        # =========================================================================
        
        n_samples = X.num_rows   # Total number of training samples
        n_features = X.num_cols  # Number of features per sample
        
        # Validate input dimensions
        if y.num_rows != n_samples:
            raise ValueError(
                f"X has {n_samples} samples but y has {y.num_rows} samples. "
                "They must match!"
            )
        if y.num_cols != 1:
            raise ValueError(
                f"y must be a column vector (n×1), but got shape {y.shape}"
            )
        
        # Validate batch size doesn't exceed dataset size
        if self.batch_size > n_samples:
            import warnings
            warnings.warn(
                f"batch_size ({self.batch_size}) is larger than dataset size ({n_samples}). "
                f"Setting batch_size to {n_samples} (equivalent to Batch GD)."
            )
            self.batch_size = n_samples
        
        # Initialize weights to zeros
        self.weights_ = Vector([0.0] * n_features)
        
        # Create list of indices for shuffling
        # We shuffle indices rather than actual data to avoid expensive copying
        indices = list(range(n_samples))
        
        # Calculate how many batches we'll have per epoch
        # Use math.ceil to handle cases where n_samples doesn't divide evenly
        # Example: 100 samples, batch_size 32 → ceil(100/32) = ceil(3.125) = 4 batches
        # The last batch will have only 100 - 3*32 = 4 samples
        num_batches = math.ceil(n_samples / self.batch_size)
        
        if self.verbose:
            print("=" * 70)
            print("STARTING MINI-BATCH GRADIENT DESCENT")
            print("=" * 70)
            print(f"Dataset: {n_samples} samples, {n_features} features")
            print(f"Batch size: {self.batch_size}")
            print(f"Batches per epoch: {num_batches}")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Epochs: {self.n_epochs}")
            print(f"Total updates: {self.n_epochs * num_batches}")
            print("-" * 70)
            print()
            print("BATCH SIZE INTERPRETATION:")
            if self.batch_size == 1:
                print("  Batch size = 1: This is equivalent to SGD!")
            elif self.batch_size == n_samples:
                print("  Batch size = n_samples: This is equivalent to Batch GD!")
            elif self.batch_size < 16:
                print("  Small batch size: More updates, noisier gradients")
            elif self.batch_size <= 64:
                print("  Standard batch size: Good balance of speed and stability")
            else:
                print("  Large batch size: Fewer updates, smoother gradients")
            print("-" * 70)
        
        # =========================================================================
        # STEP 1: THE TRAINING LOOP (EPOCH BY EPOCH)
        # =========================================================================
        
        for epoch in range(self.n_epochs):
            
            # =====================================================================
            # STEP 1A: SHUFFLE THE DATA (CRITICAL FOR MINI-BATCH GD!)
            # =====================================================================
            # Just like SGD, we shuffle every epoch to prevent the model from
            # learning patterns in the data ordering.
            #
            # But here's the key difference from SGD:
            # - SGD: Shuffle → process one by one → each sample is independent
            # - Mini-Batch: Shuffle → group into batches → samples within a batch
            #               are processed together
            #
            # This means the composition of each batch changes every epoch!
            # 
            # Example with 6 samples, batch_size=2:
            # Epoch 1: Shuffle to [3,1,5,2,4,0] → batches: [3,1], [5,2], [4,0]
            # Epoch 2: Shuffle to [0,4,2,5,1,3] → batches: [0,4], [2,5], [1,3]
            #
            # Notice: Sample 3 was with sample 1 in epoch 1, but with sample 1
            # in epoch 2. This variety helps the model learn robust patterns!
            
            random.shuffle(indices)
            
            # =====================================================================
            # STEP 1B: PROCESS DATA IN MINI-BATCHES
            # =====================================================================
            # Now we'll step through the shuffled data in chunks of size batch_size.
            #
            # The range(0, n_samples, self.batch_size) creates:
            # 0, batch_size, 2*batch_size, 3*batch_size, ..., up to n_samples
            #
            # Example with n_samples=100, batch_size=32:
            # range(0, 100, 32) → [0, 32, 64, 96]
            # So we'll create batches:
            # - Batch 1: indices[0:32] (32 samples)
            # - Batch 2: indices[32:64] (32 samples)
            # - Batch 3: indices[64:96] (32 samples)
            # - Batch 4: indices[96:100] (4 samples, the incomplete batch)
            
            for start_idx in range(0, n_samples, self.batch_size):
                
                # =============================================================
                # DETERMINE BATCH BOUNDARIES
                # =============================================================
                # The end index is either:
                # - start_idx + batch_size (for normal batches)
                # - n_samples (for the last incomplete batch)
                #
                # We use min() to handle both cases elegantly.
                
                end_idx = min(start_idx + self.batch_size, n_samples)
                
                # Extract the indices for this specific batch
                # These are the positions of samples we'll process together
                batch_indices = indices[start_idx:end_idx]
                
                # The actual batch size for this iteration
                # Usually equals self.batch_size, but might be smaller for last batch
                actual_batch_size = len(batch_indices)
                
                # HANDLING THE LAST INCOMPLETE BATCH:
                # If n_samples=100 and batch_size=32, the last batch has only 4 samples.
                # Our gradient calculation will automatically handle this because we
                # divide by the actual batch size (4) not the configured batch size (32).
                # This means the last batch contributes proportionally less to the
                # epoch's learning, which is mathematically correct!
                
                # =============================================================
                # CONSTRUCT MINI-BATCH MATRICES
                # =============================================================
                # We need to extract the rows corresponding to batch_indices from
                # both X and y to create our mini-batch.
                #
                # Ideally, we'd have efficient matrix slicing like NumPy's X[batch_indices].
                # But with our custom Matrix class, we reconstruct from lists.
                #
                # This is one area where libraries like NumPy shine — they can create
                # views of data without copying. We're doing full copies here for
                # educational clarity.
                
                # Extract rows from X for this batch
                # If batch_indices = [5, 2, 8], we get [X.elements[5], X.elements[2], X.elements[8]]
                X_batch_elements = [X.elements[i] for i in batch_indices]
                
                # Extract corresponding labels from y
                y_batch_elements = [y.elements[i] for i in batch_indices]
                
                # Construct mini-batch matrices
                X_batch = Matrix(X_batch_elements)  # Shape: (batch_size, n_features)
                y_batch = Matrix(y_batch_elements)  # Shape: (batch_size, 1)
                
                # NOW WE HAVE:
                # X_batch: A small matrix containing only the samples in this batch
                # y_batch: The corresponding labels for these samples
                #
                # Think of it as temporarily zooming in on a small subset of the data.
                # We'll compute gradients just from this subset, ignoring all other data!
                
                # =============================================================
                # FORWARD PASS: COMPUTE PREDICTIONS FOR THE BATCH
                # =============================================================
                # For each sample in the batch, compute: ŷᵢ = wᵀxᵢ
                #
                # This is vectorized in the sense that we're computing multiple
                # predictions at once (stored in a list), even though our Matrix
                # class doesn't support true vectorized operations like NumPy.
                #
                # In a real deep learning framework with GPU support, this entire
                # loop would be one matrix multiplication: y_pred = X_batch @ weights
                # and would execute in parallel on the GPU!
                
                batch_preds = []
                for r in range(X_batch.num_rows):  # Iterate through batch samples
                    # Get r-th sample in the batch as a vector
                    x_r = X_batch.row(r)
                    
                    # Compute prediction: dot product of features and weights
                    pred = x_r.dot(self.weights_)
                    
                    # Store prediction (wrapped in list for Matrix format)
                    batch_preds.append([pred])
                
                # Convert predictions to Matrix (column vector)
                y_pred_batch = Matrix(batch_preds)  # Shape: (batch_size, 1)
                
                # =============================================================
                # COMPUTE GRADIENT FROM THE MINI-BATCH
                # =============================================================
                # This is the key step! We're computing the gradient using ONLY
                # the samples in this mini-batch, not the entire dataset.
                #
                # Mathematical form:
                # ∇J_batch(w) = (1/m) Σᵢ∈batch ∇Jᵢ(w)
                #
                # where m = actual_batch_size (usually equals self.batch_size)
                #
                # This is an approximation of the true gradient:
                # ∇J_true(w) = (1/n) Σᵢ₌₁ⁿ ∇Jᵢ(w)
                #
                # The approximation quality depends on batch size:
                # - Larger batch → better approximation (closer to true gradient)
                # - Smaller batch → noisier approximation (more variance)
                #
                # But remember: noise isn't always bad! It helps exploration.
                
                gradient = self.loss_function.calculate_gradient(
                    X_batch,       # Only the batch samples
                    y_batch,       # Only the batch labels
                    y_pred_batch,  # Only the batch predictions
                    self.weights_  # Current weights (needed for some losses)
                )
                
                # The gradient is now a Vector where each component tells us how
                # to adjust that weight based on this mini-batch's average feedback.
                #
                # KEY INSIGHT: This gradient is computed from only 32 samples
                # (if batch_size=32), not all n samples! It's a noisy estimate,
                # but we'll average out the noise over many batches.
                
                # =============================================================
                # UPDATE WEIGHTS USING THE MINI-BATCH GRADIENT
                # =============================================================
                # Apply the standard gradient descent update rule:
                # w_new = w_old - α·∇J_batch(w)
                #
                # This happens ONCE PER BATCH, not once per epoch (like Batch GD)
                # and not once per sample (like SGD).
                #
                # Example timeline with 100 samples, batch_size=32:
                # - Batch 1 (samples 0-31): compute gradient → update weights
                # - Batch 2 (samples 32-63): compute gradient → update weights
                # - Batch 3 (samples 64-95): compute gradient → update weights
                # - Batch 4 (samples 96-99): compute gradient → update weights
                # → One epoch complete, made 4 weight updates!
                
                new_w_elements = []
                for j in range(n_features):
                    # For each weight wⱼ:
                    # wⱼ_new = wⱼ_old - learning_rate × gradientⱼ
                    update = self.weights_[j] - (self.learning_rate * gradient[j])
                    new_w_elements.append(update)
                
                # Replace old weights with updated weights
                self.weights_ = Vector(new_w_elements)
                
                # AT THIS POINT:
                # We've processed one mini-batch and updated the weights once.
                # The weights are now slightly different than they were before
                # this batch. The next batch will use these new weights!
                #
                # This is different from Batch GD where weights stay constant
                # throughout the entire epoch, only updating once at the end.
            
            # =====================================================================
            # STEP 1C: END OF EPOCH - OPTIONAL LOSS MONITORING
            # =====================================================================
            # We've now processed all batches in this epoch. Let's check our progress!
            #
            # IMPORTANT CONSIDERATION:
            # Computing loss on the entire dataset is expensive! For large datasets,
            # you might not want to do this every epoch. Common strategies:
            # - Compute every 10 epochs (saves time)
            # - Compute on a validation set (smaller, representative subset)
            # - Use running average of batch losses (approximate but fast)
            #
            # For learning purposes, we'll compute the full loss periodically.
            
            # Determine if we should compute loss this epoch
            should_log = (
                epoch == 0 or                                      # First epoch
                epoch == self.n_epochs - 1 or                     # Last epoch
                epoch % max(1, (self.n_epochs // 10)) == 0        # Every 10%
            )
            
            if self.verbose and should_log:
                # Compute predictions for ALL samples (not just last batch)
                # This shows us how well the model performs on the entire dataset
                full_preds = self.predict(X)
                
                # Compute loss on full dataset
                loss = self.loss_function.calculate_loss(y, full_preds)
                
                # Track in history
                self.history_['loss'].append(loss)
                self.history_['epoch'].append(epoch)
                
                # Display progress
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f} | "
                      f"Weights: {[f'{w:.4f}' for w in self.weights_.elements]}")
        
        # =========================================================================
        # STEP 2: TRAINING COMPLETE
        # =========================================================================
        
        if self.verbose:
            print("-" * 70)
            print("TRAINING COMPLETE!")
            print(f"Total parameter updates: {self.n_epochs * num_batches}")
            print(f"Final loss: {self.history_['loss'][-1]:.6f}")
            print(f"Final weights: {[f'{w:.4f}' for w in self.weights_.elements]}")
            print("=" * 70)
            print()
            print("CONVERGENCE CHARACTERISTICS:")
            print("- Mini-Batch GD converges smoother than SGD")
            print("- But with more oscillation than Batch GD")
            print("- Usually reaches 'good enough' solution quickly")
            print("- Then makes smaller refinements over remaining epochs")
            print("=" * 70)
        
        # Return self for method chaining
        return self
    
    def predict(self, X: Matrix) -> Matrix:
        """
        Make predictions using the learned weights.
        
        After training, we use the final weights to predict outputs for new data.
        The prediction process is identical for all three optimizers (Batch, SGD,
        Mini-Batch) — only the TRAINING differs!
        
        PARAMETERS:
        ----------
        X : Matrix, shape (n_samples, n_features)
            New data to make predictions for.
            
        RETURNS:
        -------
        predictions : Matrix, shape (n_samples, 1)
            Predicted values for each sample.
            
        MATHEMATICAL OPERATION:
        ----------------------
        For each sample xᵢ in X:
            ŷᵢ = w₀·xᵢ₀ + w₁·xᵢ₁ + ... + wₙ·xᵢₙ = wᵀxᵢ
        
        This is a simple dot product between the learned weights and each sample.
        
        THE WEIGHTS TELL A STORY:
        -------------------------
        Each weight represents the importance of its corresponding feature.
        
        Example: House price prediction with weights [50000, 100, 5000]
        - w₀ = 50000: Base price (bias term)
        - w₁ = 100: Each additional square foot adds $100
        - w₂ = 5000: Each additional bedroom adds $5000
        
        Prediction for a 2000 sqft, 3 bedroom house:
        ŷ = 50000 + 100(2000) + 5000(3) = 50000 + 200000 + 15000 = $265,000
        
        THE OPTIMIZER DOESN'T MATTER ANYMORE:
        ------------------------------------
        Whether you learned these weights with Batch GD (slow and steady),
        SGD (fast and chaotic), or Mini-Batch GD (balanced), once training
        is done, the weights are just numbers! The prediction process is
        exactly the same regardless of how you learned them.
        
        It's like: Once you've learned to ride a bike (whether through one
        long careful lesson, many short impulsive tries, or medium-sized
        practice sessions), the actual riding skill is the same!
        
        Example:
        -------
        >>> # After training
        >>> X_new = Matrix([[1, 2500, 3], [1, 1800, 2]])  # Two new houses
        >>> predictions = mbgd.predict(X_new)
        >>> print(predictions)  # Predicted prices for these houses
        """
        
        # Check if model has been trained
        if self.weights_ is None:
            raise Exception(
                "Model not trained yet! Call .fit(X, y) first to learn weights."
            )
        
        # Validate that new data has same number of features as training data
        if X.num_cols != len(self.weights_):
            raise ValueError(
                f"X has {X.num_cols} features but model was trained on "
                f"{len(self.weights_)} features. They must match!"
            )
        
        # Make predictions for all samples
        # For each sample xᵢ, compute ŷᵢ = wᵀxᵢ
        predictions = []
        for i in range(X.num_rows):
            # Get i-th sample as a vector
            x_i = X.row(i)
            
            # Compute dot product with weights
            y_pred_i = x_i.dot(self.weights_)
            
            # Store prediction (wrapped in list for Matrix format)
            predictions.append([y_pred_i])
        
        # Convert list of predictions to Matrix (column vector)
        return Matrix(predictions)

