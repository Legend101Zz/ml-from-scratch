"""
STOCHASTIC GRADIENT DESCENT: THE IMPULSIVE LEARNER
==================================================

This module implements Stochastic Gradient Descent (SGD), one of the most important
variations of gradient descent in modern machine learning.

THE CORE PHILOSOPHY:
-------------------
"Don't wait for perfect information — learn from every single experience immediately."

While Batch Gradient Descent waits patiently to see ALL the data before making one
careful step, SGD is like an impulsive student who adjusts their understanding after
reading EACH INDIVIDUAL example in a textbook.

REAL-WORLD ANALOGY: Learning a Language
---------------------------------------
Imagine you're learning Spanish by reading sentences:

**Batch Gradient Descent Approach:**
- Read ALL 1000 sentences in the book
- Calculate: "On average, 'gato' appeared in context X, Y, Z..."
- After processing everything, update your understanding once
- Slow but very accurate

**Stochastic Gradient Descent Approach:**
- Read ONE random sentence: "El gato es negro" (The cat is black)
- Immediately think: "Ah! 'gato' might mean cat!"
- Update your mental model RIGHT NOW
- Read another random sentence: "Mi gato come pescado" (My cat eats fish)
- Immediately reinforce: "Yes, 'gato' definitely means cat!"
- Keep going sentence by sentence, constantly adjusting

SGD is messier (you might make wrong guesses from single sentences), but you learn
MUCH faster because you're getting feedback after every single example!

THE MATHEMATICAL INTUITION:
--------------------------
In Batch GD, we compute the EXACT gradient:
    ∇J(w) = (1/n) Σᵢ₌₁ⁿ ∇Jᵢ(w)
    
This is the average gradient across ALL n samples. It points in the true downhill
direction, but requires processing all n samples before taking ONE step.

In SGD, we use a NOISY estimate:
    ∇J(w) ≈ ∇Jᵢ(w)
    
We pick ONE random sample i and use its gradient as an approximation of the true
gradient. This is WRONG on average (high variance), but:
1. It's much faster (no need to wait for all data)
2. The noise actually HELPS escape local minima (like shaking a ball in a bumpy landscape)
3. Over many iterations, the errors average out and we still reach a good solution

WHY THIS WORKS (THE MAGIC):
---------------------------
Even though each individual update is noisy and often points in a "wrong" direction,
as long as the gradient is UNBIASED (meaning it points in the right direction ON AVERAGE),
we'll eventually reach the minimum.

Think of it like this: Imagine you're blindfolded trying to reach the lowest point in
a valley. Batch GD carefully measures the slope around you before taking one perfect step.
SGD takes random steps that are ROUGHLY downhill — some steps are good, some are bad,
but on average you're making progress. You'll zigzag a lot, but you'll get there faster!

THE CRITICAL IMPORTANCE OF SHUFFLING:
-------------------------------------
Imagine your training data is organized like this:
    - Samples 1-500: All pictures of cats (label = 1)
    - Samples 501-1000: All pictures of dogs (label = 0)

If you process data in order without shuffling:
    Epoch 1:
    - Samples 1-500 (cats): Model learns "everything is a cat!"
    - Samples 501-1000 (dogs): Model forgets cats, learns "everything is a dog!"
    
    Epoch 2:
    - Repeat the same catastrophic cycle!

The model never learns the true pattern because it's stuck in a cycle of forgetting!

By shuffling, each sample becomes an independent "surprise" — the model can't memorize
the order and is forced to learn the actual underlying patterns.

WHEN TO USE SGD:
---------------
✅ Huge datasets (millions of samples) where Batch GD is too slow
✅ Online learning (data arrives one sample at a time, like web clicks)
✅ Non-convex optimization (the noise helps escape bad local minima)
✅ When you need fast initial progress (SGD converges quickly at first)

❌ When you need perfect convergence (SGD oscillates around the minimum forever)
❌ When you have very few samples (< 100) — the noise dominates
❌ When vectorization is critical (SGD can't be parallelized easily)

Author: Mrigesh
Philosophy: Learn from every experience immediately, don't wait for perfect information
"""

import random

from foundations.linear_algebra.vectors_and_matrices import Matrix, Vector

from .loss_strategies import LossFunction


class StochasticGradientDescent:
    """
    Stochastic Gradient Descent: The impulsive optimizer that learns from one sample at a time.
    
    CORE ALGORITHM:
    --------------
    For each epoch (full pass through data):
        Shuffle the dataset randomly
        For each individual sample i:
            1. Compute prediction for sample i only
            2. Compute gradient from sample i only (noisy estimate!)
            3. Update weights immediately (don't wait for other samples)
            4. Move to next random sample
    
    THE KEY DIFFERENCE FROM BATCH GD:
    ---------------------------------
    Batch GD: n samples → 1 gradient → 1 update per epoch
    SGD:      n samples → n gradients → n updates per epoch
    
    This means SGD takes n times more steps in the same amount of time!
    However, each step is based on noisy information, so the path is chaotic.
    
    THE LEARNING RATE DILEMMA:
    -------------------------
    SGD is much more sensitive to learning rate than Batch GD:
    
    - Too large: The noise causes wild oscillations, never converges
    - Too small: Takes forever to make progress
    - Just right: Fast initial progress, but never truly settles (keeps bouncing around)
    
    Common practice: Start with a reasonable learning rate (0.01), then DECAY it over
    time (e.g., multiply by 0.95 every 10 epochs). This gives you fast early progress
    and eventual convergence.
    
    THE NOISE IS A FEATURE, NOT A BUG:
    ----------------------------------
    The "drunk walk" behavior of SGD actually has benefits:
    
    1. EXPLORATION: The noise helps the optimizer explore the loss surface more thoroughly.
       If Batch GD gets stuck in a bad local minimum, it stays there forever. SGD might
       randomly jump out and find a better minimum!
    
    2. REGULARIZATION: The noise acts as implicit regularization, preventing overfitting.
       Models trained with SGD often generalize better than those trained with Batch GD!
    
    3. SPEED: For massive datasets (millions of samples), waiting for Batch GD to compute
       one perfect gradient is impractical. SGD starts improving immediately.
    
    PRACTICAL CONSIDERATIONS:
    ------------------------
    Memory: SGD processes one sample at a time, so memory usage is O(1) relative to
    dataset size. Batch GD needs O(n) memory to hold all predictions and errors.
    
    Parallelization: SGD is inherently sequential (process sample 1, then 2, then 3...).
    Modern GPUs can't parallelize this easily. This is why Mini-Batch GD is preferred
    in practice (combines SGD's speed with Batch GD's parallelization).
    
    CONVERGENCE BEHAVIOR:
    --------------------
    Batch GD: Loss decreases smoothly, monotonically, converges to a point
    SGD: Loss decreases on average but oscillates wildly, never converges to a point
    
    Think of it like this:
    - Batch GD is a ball rolling downhill that eventually stops at the bottom
    - SGD is a ball rolling downhill while being randomly kicked — it reaches the bottom
      area quickly but keeps bouncing around, never fully settling
    
    PARAMETERS:
    ----------
    loss_function : LossFunction
        The strategy for computing loss and gradients (MSE, MAE, Huber, etc.)
        
    learning_rate : float (default: 0.01)
        Step size for each update.
        CRITICAL: SGD typically needs a SMALLER learning rate than Batch GD!
        Why? Because we're updating n times per epoch instead of once.
        Rule of thumb: Use 1/10th the learning rate you'd use for Batch GD.
        
    n_epochs : int (default: 50)
        Number of complete passes through the dataset.
        Each epoch processes all n samples (in random order), making n updates.
        
    verbose : bool (default: False)
        If True, print average loss every few epochs.
        Note: We can only compute average loss AFTER seeing all samples,
        so we accumulate loss during the epoch and report at the end.
    
    ATTRIBUTES (learned during training):
    ------------------------------------
    weights_ : Vector
        The learned parameter vector after training.
        
    history_ : dict
        Training history containing:
        - 'loss': Average loss per epoch (computed after all n updates)
        - 'epoch': Epoch numbers
        
        Note: The loss will be much noisier than Batch GD's loss curve!
    """
    
    def __init__(
        self, 
        loss_function: LossFunction, 
        learning_rate: float = 0.01, 
        n_epochs: int = 50, 
        verbose: bool = False
    ):
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.verbose = verbose
        
        # These will be set during training
        self.weights_ = None
        self.history_ = {'loss': [], 'epoch': []}
    
    def fit(self, X: Matrix, y: Matrix) -> 'StochasticGradientDescent':
        """
        Fit the model using Stochastic Gradient Descent.
        
        This is where the "impulsive learning" happens! Instead of waiting to see all
        data before updating (like Batch GD), we update weights after EVERY SINGLE sample.
        
        PARAMETERS:
        ----------
        X : Matrix, shape (n_samples, n_features)
            Training data where each row is a sample, each column is a feature.
            
        y : Matrix, shape (n_samples, 1)
            Target values (labels) for each sample.
            
        RETURNS:
        -------
        self : StochasticGradientDescent
            Returns self for method chaining (allows syntax like sgd.fit(X,y).predict(X_test))
        
        THE ALGORITHM STEP BY STEP:
        ---------------------------
        For each epoch:
            1. SHUFFLE the data (critical for breaking order-dependent patterns!)
            2. For each sample i in random order:
                a. Make prediction for sample i only
                b. Compute error for sample i only
                c. Compute gradient from sample i only (noisy but fast!)
                d. Update ALL weights based on this one sample
            3. After processing all samples, compute average loss for monitoring
        
        WHY THIS IS DIFFERENT FROM BATCH GD:
        -----------------------------------
        Batch GD: One big, accurate gradient → One update
        SGD:      Many small, noisy gradients → Many updates
        
        Think of it like course correction in a self-driving car:
        - Batch GD: Wait to see the entire road ahead, calculate perfect steering angle, turn once
        - SGD: Constantly adjust steering based on what's immediately in front of you
        
        The SGD car will zigzag more, but it reacts faster to obstacles!
        
        Example:
        -------
        >>> X = Matrix([[1, 2], [1, 3], [1, 4], [1, 5]])
        >>> y = Matrix([[5], [7], [9], [11]])  # y = 1 + 2x
        >>> sgd = StochasticGradientDescent(
        ...     loss_function=MSELoss(),
        ...     learning_rate=0.001,  # Note: smaller than Batch GD!
        ...     n_epochs=100
        ... )
        >>> sgd.fit(X, y)
        >>> print(sgd.weights_)  # Should be close to [1, 2]
        """
        
        # =========================================================================
        # STEP 0: INITIALIZATION AND SETUP
        # =========================================================================
        
        n_samples = X.num_rows   # How many data points we have
        n_features = X.num_cols  # How many features per data point
        
        # Validate input dimensions (same validation as Batch GD)
        if y.num_rows != n_samples:
            raise ValueError(
                f"X has {n_samples} samples but y has {y.num_rows} samples. "
                "They must match!"
            )
        if y.num_cols != 1:
            raise ValueError(
                f"y must be a column vector (n×1), but got shape {y.shape}"
            )
        
        # Initialize weights to zeros (standard starting point)
        # Alternative: Small random values to break symmetry (used in neural networks)
        self.weights_ = Vector([0.0] * n_features)
        
        # Create a list of indices [0, 1, 2, ..., n-1] for shuffling
        # We shuffle INDICES rather than the actual data to avoid expensive copies
        indices = list(range(n_samples))
        
        if self.verbose:
            print("=" * 70)
            print("STARTING STOCHASTIC GRADIENT DESCENT")
            print("=" * 70)
            print(f"Dataset: {n_samples} samples, {n_features} features")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Epochs: {self.n_epochs}")
            print(f"Updates per epoch: {n_samples} (one per sample!)")
            print("-" * 70)
        
        # =========================================================================
        # STEP 1: THE TRAINING LOOP (EPOCH BY EPOCH)
        # =========================================================================
        
        for epoch in range(self.n_epochs):
            
            # =====================================================================
            # STEP 1A: SHUFFLE THE DATA (CRITICAL FOR SGD!)
            # =====================================================================
            # This is one of the most important lines in the entire algorithm!
            # 
            # WHY SHUFFLE?
            # -----------
            # If data is ordered (e.g., all cats, then all dogs), SGD will:
            # - See all cats → learn "predict cat for everything"
            # - See all dogs → forget cats, learn "predict dog for everything"
            # - Never converge because it's stuck in a forget-relearn cycle!
            #
            # Shuffling makes each sample a "surprise" — the model can't rely on
            # patterns in the ordering and must learn the actual data relationships.
            #
            # MATHEMATICAL JUSTIFICATION:
            # For SGD to converge, we need gradients to be UNBIASED estimates of
            # the true gradient. If we always process samples in the same order,
            # we introduce systematic bias. Random sampling removes this bias.
            
            random.shuffle(indices)
            
            # This line randomly reorders [0,1,2,3,...,n-1] so we visit samples
            # in a different order each epoch. Sample #5 might be first this epoch,
            # last the next epoch, middle the epoch after that, etc.
            
            # =====================================================================
            # STEP 1B: ACCUMULATOR FOR EPOCH LOSS
            # =====================================================================
            # We can't compute "average loss" until we've processed all samples,
            # so we accumulate loss as we go and average it at the end of the epoch.
            
            epoch_loss = 0.0
            
            # =====================================================================
            # STEP 1C: PROCESS EACH SAMPLE INDIVIDUALLY (THE HEART OF SGD!)
            # =====================================================================
            # This is where SGD differs fundamentally from Batch GD.
            # We're going to loop through EVERY SINGLE sample and update weights
            # IMMEDIATELY after seeing each one. No waiting!
            
            for i in indices:
                # We're now looking at sample number 'i' (in shuffled order)
                
                # =============================================================
                # EXTRACT SINGLE SAMPLE
                # =============================================================
                # We need to extract one row from X and one value from y.
                # BUT: Our loss function expects Matrix objects, not raw vectors.
                # So we wrap them to create a 1-row matrix.
                
                # Get the i-th row as a list (e.g., [1, 2.5, 3.1])
                x_sample_data = X.elements[i]
                y_sample_data = y.elements[i]
                
                # Wrap in a list to make it a 1×n_features matrix
                # [1, 2.5, 3.1] becomes [[1, 2.5, 3.1]]
                x_sample = Matrix([x_sample_data])  # Shape: (1, n_features)
                y_sample = Matrix([y_sample_data])  # Shape: (1, 1)
                
                # WHY DO WE NEED TO WRAP?
                # Our loss function's calculate_gradient() expects Matrix inputs
                # because it was designed to work with batches. Even though we're
                # only processing ONE sample, we format it as a "batch of size 1"
                # to maintain interface consistency.
                
                # =============================================================
                # FORWARD PASS: MAKE PREDICTION FOR THIS ONE SAMPLE
                # =============================================================
                # Compute ŷᵢ = wᵀxᵢ (dot product of weights and features)
                
                pred_val = X.row(i).dot(self.weights_)
                # This computes: w₀·x₀ + w₁·x₁ + ... + wₙ·xₙ
                # Example: If weights = [1, 2] and x_i = [1, 3], then pred = 1·1 + 2·3 = 7
                
                # Wrap prediction as a 1×1 matrix to match expected format
                y_pred = Matrix([[pred_val]])
                
                # NOW WE HAVE:
                # - x_sample: The features for sample i (as 1×n matrix)
                # - y_sample: The true label for sample i (as 1×1 matrix)
                # - y_pred: Our prediction for sample i (as 1×1 matrix)
                
                # =============================================================
                # COMPUTE GRADIENT FROM THIS SINGLE SAMPLE
                # =============================================================
                # This is the KEY difference from Batch GD!
                # 
                # Batch GD computes: ∇J = (1/n) Σᵢ ∇Jᵢ (average over ALL samples)
                # SGD computes: ∇J ≈ ∇Jᵢ (use just this ONE sample)
                #
                # This gradient is a NOISY estimate of the true gradient:
                # - Sometimes it points in the right direction
                # - Sometimes it points in a wrong direction
                # - ON AVERAGE (over many samples), it points correctly
                #
                # Think of it like learning from one quiz question vs. learning
                # from the entire exam. One question gives you a rough idea of
                # what to study, but it's not perfect information!
                
                gradient = self.loss_function.calculate_gradient(
                    x_sample,      # Features: just this one sample
                    y_sample,      # True label: just this one sample  
                    y_pred,        # Prediction: just this one sample
                    self.weights_  # Current weights (needed for some loss functions)
                )
                
                # The gradient is now a Vector of length n_features, where each
                # component tells us how to adjust that weight based on this sample.
                #
                # Example: If gradient = [0.1, -0.5], it means:
                # - Weight 0 should decrease (predicted too high for this sample)
                # - Weight 1 should increase (predicted too low for this sample)
                
                # =============================================================
                # UPDATE WEIGHTS IMMEDIATELY (DON'T WAIT!)
                # =============================================================
                # This is the "impulsive" part of SGD. We update RIGHT NOW based
                # on this one sample, without waiting to see other samples.
                #
                # Update rule: wⱼ := wⱼ - α·∇Jᵢ(w)ⱼ
                #
                # We're moving weights in the OPPOSITE direction of the gradient
                # (because gradient points uphill, we want to go downhill).
                
                new_w_elements = []
                for j in range(n_features):
                    # For each weight wⱼ:
                    # 1. Take current value: self.weights_[j]
                    # 2. Subtract learning_rate × gradient[j]
                    # 3. Store the new value
                    
                    update = self.weights_[j] - (self.learning_rate * gradient[j])
                    new_w_elements.append(update)
                    
                    # LEARNING RATE CONSIDERATION:
                    # We're updating n times per epoch (once per sample).
                    # Batch GD updates once per epoch.
                    # So if we use the same learning rate, SGD effectively takes
                    # n times bigger steps! That's why SGD needs smaller learning rates.
                
                # Replace old weights with new weights
                self.weights_ = Vector(new_w_elements)
                
                # AT THIS POINT: We've updated the model based on ONE sample!
                # The next sample we see will use these updated weights.
                # This is like learning Spanish sentence by sentence, updating
                # your vocabulary after each sentence, rather than waiting to
                # read the whole book before updating anything.
                
                # =============================================================
                # ACCUMULATE LOSS FOR MONITORING
                # =============================================================
                # We can't report "average loss" until we've seen all samples,
                # so we just add this sample's loss to a running total.
                
                sample_loss = self.loss_function.calculate_loss(y_sample, y_pred)
                epoch_loss += sample_loss
                
                # Note: This loss was computed BEFORE we updated weights, so it
                # reflects how well the model performed on this sample with the
                # OLD weights. After updating, the model is slightly different.
            
            # =====================================================================
            # STEP 1D: END OF EPOCH - COMPUTE AVERAGE LOSS
            # =====================================================================
            # We've now processed all n samples (in random order), making n updates.
            # Let's compute the average loss across all samples for this epoch.
            
            avg_loss = epoch_loss / n_samples
            
            # This loss represents: "How well did the model perform on all samples
            # using the various weight configurations throughout this epoch?"
            #
            # Note: Because weights changed after EVERY sample, this is actually
            # an average over n different model configurations! This is why SGD's
            # loss curve is much noisier than Batch GD's.
            
            # =====================================================================
            # STEP 1E: LOGGING AND HISTORY TRACKING
            # =====================================================================
            
            # Always track history (even if not verbose) so users can plot later
            self.history_['loss'].append(avg_loss)
            self.history_['epoch'].append(epoch)
            
            # Print progress if verbose (every 10% of training)
            if self.verbose:
                should_print = (
                    epoch == 0 or                                    # First epoch
                    epoch == self.n_epochs - 1 or                   # Last epoch
                    epoch % max(1, (self.n_epochs // 10)) == 0     # Every 10%
                )
                
                if should_print:
                    print(f"Epoch {epoch:4d} | Avg Loss: {avg_loss:.6f} | "
                          f"Weights: {[f'{w:.4f}' for w in self.weights_.elements]}")
        
        # =========================================================================
        # STEP 2: TRAINING COMPLETE
        # =========================================================================
        
        if self.verbose:
            print("-" * 70)
            print("TRAINING COMPLETE!")
            print(f"Total parameter updates: {n_samples * self.n_epochs}")
            print(f"Final loss: {self.history_['loss'][-1]:.6f}")
            print(f"Final weights: {[f'{w:.4f}' for w in self.weights_.elements]}")
            print("=" * 70)
            print()
            print("EXPECTED BEHAVIOR:")
            print("- Loss should decrease on average (but with oscillations)")
            print("- Convergence is not as smooth as Batch GD")
            print("- Final loss might be slightly higher than Batch GD")
            print("  (because SGD never truly converges, it keeps bouncing)")
            print("=" * 70)
        
        # Return self to allow method chaining: sgd.fit(X, y).predict(X_test)
        return self
    
    def predict(self, X: Matrix) -> Matrix:
        """
        Make predictions using the learned weights.
        
        After training with SGD, we can use the final weights to predict outputs
        for new data. The prediction process is identical to Batch GD — only the
        TRAINING process differs.
        
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
        For each sample xᵢ:
            ŷᵢ = w₀·xᵢ₀ + w₁·xᵢ₁ + ... + wₙ·xᵢₙ = wᵀxᵢ
        
        This is just a dot product — the same operation we did during training!
        
        IMPORTANT NOTE:
        --------------
        Even though SGD training is noisy and chaotic, the FINAL weights we end up
        with are a good solution! The weights don't remember "how" they were learned,
        only their final values matter.
        
        Think of it like this: Whether you learned Spanish by studying one sentence
        at a time (SGD) or reading whole chapters (Batch GD), once you've learned it,
        you speak Spanish equally well!
        
        Example:
        -------
        >>> # After training on house data
        >>> X_new = Matrix([[1, 2500], [1, 1800]])  # Two new houses
        >>> predictions = sgd.predict(X_new)
        >>> print(predictions)  # Predicted prices
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
        
        # Make predictions: ŷ = X @ w (matrix-vector multiplication)
        predictions = []
        for i in range(X.num_rows):
            # Get i-th sample as a vector
            x_i = X.row(i)
            
            # Compute dot product: ŷᵢ = xᵢᵀw
            y_pred_i = x_i.dot(self.weights_)
            
            # Store as a list (to create column vector later)
            predictions.append([y_pred_i])
        
        # Convert list of predictions to Matrix (column vector)
        return Matrix(predictions)
