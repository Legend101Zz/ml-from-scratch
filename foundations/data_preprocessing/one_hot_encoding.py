"""
ONE-HOT ENCODING: TRANSFORMING CATEGORICAL DATA INTO NUMBERS
============================================================

Machine learning algorithms work with numbers, not categories. But much of real-world
data is categorical: colors, cities, product types, user preferences. One-hot encoding
solves this fundamental problem by converting categories into binary vectors.

TEACHING PHILOSOPHY:
-------------------
We'll understand one-hot encoding from multiple perspectives:
1. MATHEMATICAL: Binary vector representation
2. GEOMETRIC: Points in high-dimensional space
3. PRACTICAL: When and how to use it correctly

THE FUNDAMENTAL PROBLEM:
-----------------------
Imagine you have a dataset of houses with a "city" feature:
- House 1: city = "New York"
- House 2: city = "Los Angeles"
- House 3: city = "Chicago"

You cannot feed strings into a model! You need numbers. Naive approaches fail:

❌ **Label Encoding**: New York=0, LA=1, Chicago=2
   Problem: This implies an ordering! It suggests LA is "between" NY and Chicago,
   and that Chicago is "twice as far" from NY as LA is. This is meaningless for
   categorical data and will confuse the model.

✓ **One-Hot Encoding**: Create a binary column for each category
   New York: [1, 0, 0]
   LA:       [0, 1, 0]
   Chicago:  [0, 0, 1]
   
   Now each city is a unique direction in space with no implied ordering!

MATHEMATICAL REPRESENTATION:
----------------------------
For a categorical variable with K unique values, we create K binary features:

Original: x ∈ {category_1, category_2, ..., category_K}
Encoded:  x → [x_1, x_2, ..., x_K] where x_i ∈ {0, 1}

Exactly one x_i equals 1 (the "hot" position), all others are 0.

This is called "one-hot" because exactly one bit is "hot" (on), others are "cold" (off).

GEOMETRIC INTERPRETATION:
-------------------------
Each category becomes a vertex of a unit hypercube!

For 3 categories:
- Category A → [1, 0, 0] (vertex at x-axis)
- Category B → [0, 1, 0] (vertex at y-axis)
- Category C → [0, 0, 1] (vertex at z-axis)

All categories are equidistant from each other: distance = √2
This treats all categories equally with no implied similarity!

THE DUMMY VARIABLE TRAP:
------------------------
⚠️ Critical Issue: If you use one-hot encoded features with an intercept term
in linear models, you create perfect multicollinearity!

Example: If you know it's NOT New York and NOT LA, it MUST be Chicago.
The three binary features sum to 1, so they're linearly dependent.

**Solution**: Drop one category (the "reference" category)
- Keep: NY=1/0, LA=1/0
- Drop: Chicago (implied when both others are 0)

This is called "dummy encoding" or "K-1 encoding."

When to drop a column:
✓ Linear regression (need to avoid multicollinearity)
✓ Logistic regression (same reason)
✗ Decision trees (don't care about multicollinearity)
✗ Neural networks (can learn despite redundancy, sometimes helps)

ORDINAL VS NOMINAL CATEGORIES:
------------------------------
**NOMINAL** (no order): colors, cities, product types
→ Use one-hot encoding

**ORDINAL** (natural order): shirt sizes (S < M < L < XL), education level
→ Consider label encoding (0, 1, 2, 3) to preserve order
→ Or one-hot if you want model to learn relationships from scratch

HANDLING UNSEEN CATEGORIES:
---------------------------
What if training has {A, B, C} but test data has category D?

**Option 1**: Encode as all zeros [0, 0, 0]
   Pro: Simple, model can continue
   Con: Lumps all unknown categories together

**Option 2**: Raise an error
   Pro: Alerts you to data shift
   Con: Prevents prediction

**Option 3**: Add "Other" category during training
   Pro: Model learns to handle unknowns
   Con: Need domain knowledge to identify rare categories

SPARSE REPRESENTATION:
---------------------
For many categories (e.g., 10,000 words in NLP), one-hot vectors are mostly zeros!

Dense: [0, 0, 0, ..., 1, ..., 0, 0, 0] (memory intensive!)
Sparse: Store only position of the 1 (memory efficient!)

Professional libraries use sparse matrices for efficiency.

MACHINE LEARNING APPLICATIONS:
-----------------------------

1. TABULAR DATA:
   - User demographics (gender, country, occupation)
   - Product categories (electronics, clothing, food)
   - Transaction types (credit, debit, cash)

2. NATURAL LANGUAGE PROCESSING:
   - Bag-of-words: each word is a category, encode as one-hot
   - But: vocabulary can be huge (100k+ words!)
   - Modern solution: word embeddings (learned dense vectors)

3. COMPUTER VISION:
   - Class labels in classification (cat, dog, bird)
   - Output layer of classifier: one-hot encoded targets

4. RECOMMENDATION SYSTEMS:
   - User IDs, item IDs as categorical features
   - Usually combined with embeddings for efficiency

ALTERNATIVES TO ONE-HOT:
-----------------------

**Target Encoding**: Replace category with mean of target variable
- Pro: Single numerical feature, no dimension explosion
- Con: Requires target variable (supervised), prone to overfitting

**Embeddings** (Neural Networks): Learn dense representations
- Pro: Lower dimensional, captures similarities
- Con: Requires training, more complex

**Feature Hashing**: Hash categories to fixed number of bins
- Pro: Handles unlimited categories, memory efficient
- Con: Collisions (different categories → same bin)

When to use each:
- One-Hot: Small number of categories (<50), interpretability matters
- Embeddings: Large number of categories, neural networks
- Target Encoding: Tree-based models, careful validation
- Hashing: Streaming data, unknown vocabulary size

Let's implement one-hot encoding properly!
"""

from typing import Any, Dict, List, Optional, Union

from foundations.linear_algebra.vectors_and_matrices import Matrix
from foundations.statistics.descriptive import *


class OneHotEncoder:
    """
    Encode categorical features as one-hot numeric arrays.
    
    This encoder transforms each categorical feature with K categories into
    K binary features, with exactly one "hot" (1) and others "cold" (0).
    
    ATTRIBUTES:
    ----------
    categories_ : List[List[Any]] or None
        The categories found for each feature during fit.
        categories_[i] holds the unique categories for feature i.
        
    drop : Optional[str or List], default=None
        Whether to drop one category per feature to avoid multicollinearity.
        - None: Keep all categories (K features per categorical variable)
        - 'first': Drop first category of each feature
        - 'if_binary': Drop first category only for binary features
        - List: Specify which category to drop for each feature
        
    handle_unknown : str, default='error'
        How to handle unknown categories during transform:
        - 'error': Raise error if unknown category appears
        - 'ignore': Encode unknown categories as all zeros
        
    sparse_output : bool, default=False
        Whether to return sparse matrix (efficient for many categories)
        We'll implement dense version for learning, note sparse option exists.
    
    n_features_in_ : int or None
        Number of features seen during fit
        
    feature_names_in_ : List[str] or None
        Names of features (if provided)
    
    DESIGN NOTES:
    ------------
    We follow sklearn's API for compatibility:
    - fit() learns categories from training data
    - transform() converts categories to one-hot vectors
    - fit_transform() does both
    - inverse_transform() converts back to original categories
    
    WHY STORE CATEGORIES:
    --------------------
    We must use the SAME encoding for train and test data!
    
    Training categories: {Red, Blue, Green}
    Test categories: {Red, Blue, Yellow}
    
    We encode test data using training categories only:
    - Red → [1, 0, 0]
    - Blue → [0, 1, 0]
    - Yellow → [0, 0, 0] (unknown) or error
    
    This consistency is crucial for model predictions.
    
    Example:
    -------
    >>> X_train = Matrix([['Red'], ['Blue'], ['Green'], ['Red']])
    >>> X_test = Matrix([['Blue'], ['Green']])
    >>> 
    >>> encoder = OneHotEncoder()
    >>> encoder.fit(X_train)
    >>> X_train_encoded = encoder.transform(X_train)
    >>> # X_train_encoded:
    >>> # [[1, 0, 0],  # Red
    >>> #  [0, 1, 0],  # Blue
    >>> #  [0, 0, 1],  # Green
    >>> #  [1, 0, 0]]  # Red
    >>> 
    >>> X_test_encoded = encoder.transform(X_test)
    >>> # Uses same encoding learned from training!
    """
    def __init__(self,
                 drop: Optional[Union[str,List]] = None,
                 handle_unknown: str = 'error',
                 sparse_output: bool = False
                 ):
         if handle_unknown not in ['error', 'ignore']:
            raise ValueError(f"handle_unknown must be 'error' or 'ignore', got {handle_unknown}")
         self.drop = drop
         self.handle_unknown = handle_unknown
         self.sparse_output = sparse_output
         
         self.categories: Optional[List[List[Any]]] = None
         self.n_features_in_: Optional[int] = None
         self._drop_idx: Optional[List[Optional[int]]] = None
         
    def fit(self, X: Matrix, y=None) -> 'OneHotEncoder':
         """
         Fit OneHotEncoder to X.
         This learns the categories present in each feature of the training data.
         """
         if not isinstance(X, Matrix):
               raise TypeError("Input X must be a Matrix object.")
         
         self.n_features_in_ = X.num_cols
         self.categories_ = []
         for i in range(self.n_features_in_):
               # Get unique categories in this feature column and sort them
               categories = sorted(list(set(X.column(i))))
               self.categories_.append(categories)
         
         self._compute_drop_idx()
         return self
      
    def transform(self, X: Matrix) -> Union[Matrix, List[Dict[int, float]]]:
         """Transform X using one-hot encoding."""
         if self.categories_ is None:
               raise ValueError("This OneHotEncoder instance is not fitted yet.")
         if X.num_cols != self.n_features_in_:
               raise ValueError(f"X has {X.num_cols} features, but encoder expects {self.n_features_in_}.")

         dense_rows = []
         sparse_rows = []
         
         for i in range(X.num_rows):
               current_dense_row_parts = []
               current_sparse_row = {}
               output_col_offset = 0
               
               for j in range(self.n_features_in_):
                  categories = self.categories_[j]
                  value = X.elements[i][j]
                  
                  # Create base binary vector
                  binary_vector = [0.0] * len(categories)
                  cat_index = -1
                  
                  try:
                     cat_index = categories.index(value)
                     binary_vector[cat_index] = 1.0
                  except ValueError: # Unknown category
                     if self.handle_unknown == 'error':
                           raise ValueError(f"Unknown category '{value}' in feature {j}.")
                     # If 'ignore', vector remains all zeros

                  # Apply drop
                  drop_idx = self._drop_idx[j] if self._drop_idx else None
                  if drop_idx is not None:
                     final_binary_vector = binary_vector[:drop_idx] + binary_vector[drop_idx+1:]
                  else:
                     final_binary_vector = binary_vector

                  # Store in appropriate format
                  if self.sparse_output:
                     for k, val in enumerate(final_binary_vector):
                           if val == 1.0:
                              current_sparse_row[output_col_offset + k] = 1.0
                  else:
                     current_dense_row_parts.extend(final_binary_vector)
                  
                  output_col_offset += len(final_binary_vector)
               
               if self.sparse_output:
                  sparse_rows.append(current_sparse_row)
               else:
                  dense_rows.append(current_dense_row_parts)

         if self.sparse_output:
               return sparse_rows
         else:
               return Matrix(dense_rows)

    def fit_transform(self, X: Matrix, y=None) -> Union[Matrix, List[Dict[int, float]]]:
      """Fit to data, then transform it."""
      return self.fit(X).transform(X)

    def inverse_transform(self, X_encoded: Union[Matrix, List[Dict[int, float]]]) -> Matrix:
      """Convert one-hot encoded data back to original categories."""
      if self.categories_ is None:
            raise ValueError("Encoder has not been fitted yet")

      if isinstance(X_encoded, Matrix):
            return self._inverse_transform_dense(X_encoded)
      elif isinstance(X_encoded, list):
            return self._inverse_transform_sparse(X_encoded)
      else:
            raise TypeError("Input must be a Matrix or a list of dictionaries (sparse format)")
         
    def get_feature_names(self, input_features: Optional[List[str]] = None) -> List[str]:
      """Get output feature names for transformation."""
      if self.categories_ is None:
            raise ValueError("Encoder has not been fitted yet")
      
      if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
      
      feature_names = []
      for i, categories in enumerate(self.categories_):
            drop_idx = self._drop_idx[i] if self._drop_idx else None
            for j, category in enumerate(categories):
               if j == drop_idx:
                  continue
               feature_names.append(f"{input_features[i]}_{category}")
      return feature_names

    def _inverse_transform_dense(self, X_dense: Matrix) -> Matrix:
      """Helper for dense matrix inverse transform."""
      X_original_rows = []
      for i in range(X_dense.num_rows):
            current_original_row = []
            col_offset = 0
            for j in range(self.n_features_in_):
               cats = self.categories_[j]
               n_cats = len(cats)
               drop_idx = self._drop_idx[j] if self._drop_idx else None
               n_cols_out = n_cats - 1 if drop_idx is not None else n_cats
               
               feature_vec = X_dense.elements[i][col_offset : col_offset + n_cols_out]
               
               hot_index_in_block = -1
               try:
                  hot_index_in_block = feature_vec.index(1.0)
               except ValueError:
                  pass # All zeros

               # Map back to original category index
               original_cat_index = 0 # Default for unknown
               if hot_index_in_block == -1: # All zeros
                  original_cat_index = drop_idx if drop_idx is not None else 0
               else: # Found a 1
                  if drop_idx is None:
                        original_cat_index = hot_index_in_block
                  else:
                        original_cat_index = hot_index_in_block + 1 if hot_index_in_block >= drop_idx else hot_index_in_block
               
               current_original_row.append(cats[original_cat_index])
               col_offset += n_cols_out
            X_original_rows.append(current_original_row)
      return Matrix(X_original_rows)

    def _inverse_transform_sparse(self, X_sparse: List[Dict[int, float]]) -> Matrix:
      """Helper for sparse list-of-dicts inverse transform."""
      X_original_rows = []
      for sparse_row in X_sparse:
            current_original_row = []
            col_offset = 0
            for j in range(self.n_features_in_):
               cats = self.categories_[j]
               n_cats = len(cats)
               drop_idx = self._drop_idx[j] if self._drop_idx else None
               n_cols_out = n_cats - 1 if drop_idx is not None else n_cats
               
               hot_index_in_block = -1
               for k in range(n_cols_out):
                  if sparse_row.get(col_offset + k) == 1.0:
                        hot_index_in_block = k
                        break

               # Map back to original category index (same logic as dense)
               original_cat_index = 0
               if hot_index_in_block == -1:
                  original_cat_index = drop_idx if drop_idx is not None else 0
               else:
                  if drop_idx is None:
                        original_cat_index = hot_index_in_block
                  else:
                        original_cat_index = hot_index_in_block + 1 if hot_index_in_block >= drop_idx else hot_index_in_block
                        
               current_original_row.append(cats[original_cat_index])
               col_offset += n_cols_out
            X_original_rows.append(current_original_row)
      return Matrix(X_original_rows)
   
    def _compute_drop_idx(self):
      """Compute which category to drop for each feature."""
      if self.drop is None or self.categories_ is None:
            self._drop_idx = None
            return
            
      self._drop_idx = []
      for i in range(self.n_features_in_):
            categories = self.categories_[i]
            n_categories = len(categories)
            
            if self.drop == 'first':
               self._drop_idx.append(0)
            elif self.drop == 'if_binary' and n_categories == 2:
               self._drop_idx.append(0)
            elif isinstance(self.drop, list):
               try:
                  idx = categories.index(self.drop[i])
                  self._drop_idx.append(idx)
               except (ValueError, IndexError):
                  self._drop_idx.append(None) # Category not found or list too short
            else:
               self._drop_idx.append(None)



   
# ==============================================================================
# USAGE EXAMPLES AND TESTS
# ==============================================================================

if __name__ == "__main__": 
    print("=" * 70)
    print("ONE-HOT ENCODING: INTERACTIVE TUTORIAL (using custom Matrix)")
    print("=" * 70)
    
    print("\n### BASIC ONE-HOT ENCODING ###\n")
    
    X_colors = Matrix([
        ['Red'],
        ['Blue'],
        ['Green'],
        ['Red'],
        ['Blue']
    ])
    
    print("Original data (colors):")
    print([row[0] for row in X_colors.elements])
    
    encoder = OneHotEncoder()
    encoder.fit(X_colors)
    X_encoded = encoder.transform(X_colors)
    
    print(f"\nLearned categories: {encoder.categories_[0]}")
    print("\nOne-hot encoded (dense Matrix):")
    print(X_encoded)
    print("\nEach row: [Blue, Green, Red]")
    
    print("\n### MULTIPLE FEATURES ###\n")
    
    X_multi = Matrix([
        ['Red',   'Small'],
        ['Blue',  'Large'],
        ['Green', 'Small'],
        ['Red',   'Medium']
    ])
    
    print("Original data (color and size):")
    print(X_multi)
    
    encoder_multi = OneHotEncoder()
    X_multi_encoded = encoder_multi.fit_transform(X_multi)
    
    print(f"\nCategories per feature:")
    print(f"  Color: {encoder_multi.categories_[0]}")
    print(f"  Size:  {encoder_multi.categories_[1]}")
    
    print(f"\nOne-hot encoded shape: {X_multi_encoded.shape}")
    print(X_multi_encoded)
    
    feature_names = encoder_multi.get_feature_names(['color', 'size'])
    print(f"\nFeature names: {feature_names}")

    print("\n### SPARSE OUTPUT (NEW!) ###\n")

    encoder_sparse = OneHotEncoder(sparse_output=True)
    X_sparse_encoded = encoder_sparse.fit_transform(X_multi)
    print("Encoded as sparse list-of-dictionaries:")
    for row in X_sparse_encoded:
        print(row)
    print("\n→ This is much more memory efficient for many categories!")
    
    print("\n### THE DUMMY VARIABLE TRAP (drop='first') ###\n")
    
    encoder_drop = OneHotEncoder(drop='first')
    X_dropped = encoder_drop.fit_transform(X_multi)
    feature_names_dropped = encoder_drop.get_feature_names(['color', 'size'])
    
    print(f"\nWith drop='first', shape: {X_dropped.shape}")
    print(f"Remaining features: {feature_names_dropped}")
    print(X_dropped)
    
    print("\n### HANDLING UNKNOWN CATEGORIES ###\n")
    
    X_train = Matrix([['Cat'], ['Dog'], ['Bird']])
    X_test = Matrix([['Dog'], ['Rabbit']]) # Rabbit not in training!
    
    print("Training data:", [row[0] for row in X_train.elements])
    print("Test data:", [row[0] for row in X_test.elements])
    
    # Strategy 1: Raise error
    encoder_error = OneHotEncoder(handle_unknown='error')
    encoder_error.fit(X_train)
    print("\nWith handle_unknown='error':")
    try:
        encoder_error.transform(X_test)
    except ValueError as e:
        print(f"✗ Error raised: {e}")

    # Strategy 2: Ignore (encode as zeros)
    encoder_ignore = OneHotEncoder(handle_unknown='ignore')
    encoder_ignore.fit(X_train)
    X_test_encoded = encoder_ignore.transform(X_test)
    print("\nWith handle_unknown='ignore':")
    print(X_test_encoded)
    print("→ Rabbit encoded as [0.0, 0.0, 0.0] (all zeros)")

    print("\n### INVERSE TRANSFORM (from Dense) ###\n")
    
    X_original = Matrix([['Apple'], ['Banana'], ['Orange']])
    encoder_inv = OneHotEncoder()
    X_encoded_dense = encoder_inv.fit_transform(X_original)
    X_reconstructed = encoder_inv.inverse_transform(X_encoded_dense)
    
    print("Original:")
    print(X_original)
    print("\nReconstructed:")
    print(X_reconstructed)
    print(f"\nPerfect reconstruction? {X_original.elements == X_reconstructed.elements}")

    print("\n### INVERSE TRANSFORM (from Sparse) ###\n")
    
    encoder_inv_sparse = OneHotEncoder(sparse_output=True)
    X_encoded_sparse = encoder_inv_sparse.fit_transform(X_original)
    X_reconstructed_sparse = encoder_inv_sparse.inverse_transform(X_encoded_sparse)

    print("Original:")
    print(X_original)
    print("\nSparse encoded:")
    for row in X_encoded_sparse:
        print(row)
    print("\nReconstructed from sparse:")
    print(X_reconstructed_sparse)
    print(f"\nPerfect reconstruction? {X_original.elements == X_reconstructed_sparse.elements}")
    
    print("\n✅ One-hot encoding (dense and sparse) complete!")