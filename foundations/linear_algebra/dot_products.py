"""
DOT PRODUCTS: THE FUNDAMENTAL SIMILARITY MEASURE
================================================

The dot product is perhaps the single most important operation in all of machine learning.
It appears literally everywhere: in every prediction, every gradient, every similarity measure.

This module explores the dot product from multiple angles:
1. Algebraic: How to compute it
2. Geometric: What it means visually
3. Statistical: How it measures correlation
4. Applied: How it appears in real ML algorithms

-------------------
We will not just implement dot products - we will understand WHY they exist and HOY they
power machine learning. By the end, you will recognize dot products hiding in algorithms
and understand what they are really computing.

THE BIG IDEA:
------------
The dot product measures ALIGNMENT between vectors. It answers the fundamental question:
"How much do these two vectors point in the same direction?"

This simple idea powers:
- Linear regression (weights · features = prediction)
- Neural networks (every neuron computes a dot product)
- Cosine similarity (measuring document similarity)
- Attention mechanisms (which words are relevant?)
- Kernel methods (similarity in transformed spaces)

REAL-WORLD ANALOGY:
------------------
Imagine two people ranking their favorite fruits from 1 to 5.
Person A: [apples: 5, oranges: 1, bananas: 3]
Person B: [apples: 4, oranges: 2, bananas: 3]

The dot product is: 5*4 + 1*2 + 3*3 = 20 + 2 + 9 = 31

What does this number mean? It measures how much their preferences agree!
- If both love apples (both give high ratings), this contributes a lot
- If one loves oranges and the other hates them, this contributes little
- The total tells you: do they have similar taste?

This is exactly how recommendation systems work!
"""

import math
from typing import List, Tuple

from vectors_and_matrices import \
    Vector  # see how what we just created is getting reused , that is why programming is so powerful

# ==============================================================================
# CORE DOT PRODUCT OPERATIONS
# ==============================================================================

def dot_product(v1: Vector, v2: Vector) -> float:
    """
    Compute the dot product (inner product, scalar product) of two vectors.
    
    This is the most fundamental operation in linear algebra for machine learning.
    
    Parameters:
    ----------
    v1, v2 : Vector
        The two vectors to compute the dot product of
        Must have the same dimension!
    
    Returns:
    -------
    float : The dot product (a single number, not a vector)
    
    MATHEMATICAL DEFINITION:
    -----------------------
    For vectors a = [a1, a2, ..., an] and b = [b1, b2, ..., bn]:
    
    a · b = a1*b1 + a2*b2 + ... + an*bn
    
    Simply: multiply corresponding components, then sum them up.
    
    PROPERTIES OF DOT PRODUCT:
    -------------------------
    1. COMMUTATIVE: a · b = b · a
       Order doesn't matter! This is because multiplication is commutative.
    
    2. DISTRIBUTIVE: a · (b + c) = a · b + a · c
       You can distribute across addition.
    
    3. SCALAR MULTIPLICATION: (ca) · b = c(a · b) = a · (cb)
       Scalars can be pulled out.
    
    4. POSITIVE DEFINITE: v · v ≥ 0, and v · v = 0 only if v = 0
       A vector dotted with itself gives its squared length.
    
    GEOMETRIC INTERPRETATION:
    ------------------------
    The dot product is intimately connected to the angle θ between vectors:
    
    a · b = ||a|| * ||b|| * cos(θ)
    
    Where ||a|| means the length (norm) of vector a.
    
    This formula reveals deep geometric meaning:
    
    - If θ = 0° (parallel): cos(0) = 1, so a · b = ||a|| * ||b|| (MAXIMUM)
      Vectors pointing the same way have maximum positive dot product.
    
    - If θ = 90° (perpendicular): cos(90°) = 0, so a · b = 0
      Orthogonal vectors have ZERO dot product. No alignment!
    
    - If θ = 180° (opposite): cos(180°) = -1, so a · b = -||a|| * ||b|| (MINIMUM)
      Vectors pointing opposite ways have maximum negative dot product.
    
    PROJECTION INTERPRETATION:
    -------------------------
    The dot product also relates to PROJECTION:
    
    (a · b) / ||b|| = length of projection of a onto b
    
    This measures "how much of vector a points in the direction of vector b."
    Imagine shining a flashlight perpendicular to vector b, casting a's shadow onto b.
    The length of that shadow is exactly (a · b) / ||b||.
    
    STATISTICAL INTERPRETATION:
    --------------------------
    For data vectors (with features as components), the dot product is related to COVARIANCE.
    
    If x and y are data vectors centered to have mean zero, then:
    (x · y) / n is the sample covariance between them
    
    Positive dot product → variables tend to increase together (positive correlation)
    Zero dot product → variables are uncorrelated
    Negative dot product → one increases when the other decreases (negative correlation)
    
    MACHINE LEARNING APPLICATIONS:
    -----------------------------
    
    1. LINEAR MODELS:
       In linear regression, the prediction is: y = w · x + b
       where w is the weight vector and x is the feature vector.
       The dot product computes a weighted sum of features!
    
    2. NEURAL NETWORKS:
       Every neuron computes: output = activation(w · x + b)
       The dot product aggregates inputs according to learned weights.
       A deep network performs billions of dot products during forward pass!
    
    3. COSINE SIMILARITY:
       similarity = (a · b) / (||a|| * ||b||) = cos(θ)
       This measures how similar two vectors are by their direction, ignoring magnitude.
       Used heavily in NLP to compare documents, in recommendation systems, etc.
    
    4. ATTENTION MECHANISMS:
       In transformers, attention scores are computed as:
       attention = softmax(Q · K^T)
       Each element is a dot product measuring relevance!
    
    5. KERNELS IN SVM:
       The kernel trick replaces dot products in the original space with dot products
       in a transformed space: K(x, y) = φ(x) · φ(y)
       This allows nonlinear decision boundaries using linear math!
    
    WHY IT'S CALLED "DOT" PRODUCT:
    ------------------------------
    The notation a · b uses a dot symbol, hence "dot product."
    Also called:
    - Inner product (because it takes two vectors and produces a scalar "inside" them)
    - Scalar product (because the result is a scalar, not a vector)
    
    COMPUTATIONAL EFFICIENCY:
    ------------------------
    Computing a dot product requires:
    - n multiplications (one per component)
    - n-1 additions (summing n products)
    - Total: O(n) operations, linear in dimension
    
    For large vectors (millions of dimensions in deep learning), this can be expensive,
    but modern hardware (GPUs) can parallelize these operations extremely well.
    
    Example:
    -------
    >>> v1 = Vector([1, 2, 3])
    >>> v2 = Vector([4, 5, 6])
    >>> dot_product(v1, v2)
    32.0  # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    
    Intuition: v1 and v2 have positive components, and they point in similar directions
    (all positive), so their dot product is positive and fairly large.
    
    >>> v_perpendicular1 = Vector([1, 0])
    >>> v_perpendicular2 = Vector([0, 1])
    >>> dot_product(v_perpendicular1, v_perpendicular2)
    0.0  # They're perpendicular (90 degrees apart)!
    """
    # We can use the dot() method we already implemented in the Vector class , lol just wanted to give the detailed explaination above
    return v1.dot(v2)

def cosine_similarity(v1: Vector, v2: Vector) -> float:
    """
    Compute cosine similarity: measures angle between vectors (direction, not magnitude).
    
    Parameters:
    ----------
    v1, v2 : Vector
        The two vectors to compare
    
    Returns:
    -------
    float : Cosine similarity, always in range [-1, 1]
            1.0 means identical direction
            0.0 means perpendicular (90 degrees)
           -1.0 means opposite direction (180 degrees)
    
    MATHEMATICAL DEFINITION:
    -----------------------
    cosine_similarity(a, b) = (a · b) / (||a|| * ||b||) = cos(θ)
    
    Where θ is the angle between the vectors.
    
    WHY THIS IS USEFUL:
    ------------------
    Regular dot product depends on both direction AND magnitude.
    If you double a vector's length, its dot product with any other vector doubles too.
    
    But sometimes you only care about DIRECTION:
    - In document similarity, a long document and a short one might be about the same topic
    - In recommendation systems, one user might rate everything highly, another lowly,
      but they might have similar preferences (just different rating scales)
    
    Cosine similarity normalizes out the magnitude, leaving only directional comparison.
    
    GEOMETRIC INTERPRETATION:
    ------------------------
    This is literally the cosine of the angle between vectors!
    
    - cos(0°) = 1.0: vectors point the same way (maximally similar)
    - cos(45°) = 0.707: vectors are somewhat similar
    - cos(90°) = 0.0: vectors are perpendicular (independent)
    - cos(135°) = -0.707: vectors point somewhat opposite ways
    - cos(180°) = -1.0: vectors point exactly opposite (maximally dissimilar)
    
    MACHINE LEARNING APPLICATIONS:
    -----------------------------
    
    1. TEXT SIMILARITY:
       Represent documents as word count vectors. Two documents are similar if their
       word count vectors have small angle between them, even if one is much longer.
       
       Example:
       Doc A: "cat dog cat" → [2, 1, 0, 0, ...] (cat count, dog count, ...)
       Doc B: "cat dog" → [1, 1, 0, 0, ...]
       
       Regular dot product: 2*1 + 1*1 = 3
       But this is affected by document length!
       
       Cosine similarity: (2*1 + 1*1) / (sqrt(5) * sqrt(2)) ≈ 0.948
       High similarity! They have the same relative word frequencies.
    
    2. RECOMMENDATION SYSTEMS:
       Find similar users by comparing their rating vectors.
       Cosine similarity is robust to different rating scales.
    
    3. FACE RECOGNITION:
       Compare face embedding vectors (from deep networks).
       Similar faces have embeddings with small angle between them.
    
    4. WORD EMBEDDINGS:
       In Word2Vec or GloVe, similar words have similar embedding vectors.
       Cosine similarity measures semantic similarity.
       Example: cos(king - man + woman, queen) is very high!
    
    WHY NOT JUST USE DOT PRODUCT?
    -----------------------------
    Consider these examples:
    
    Example 1:
    a = [1, 1] (length √2)
    b = [2, 2] (length 2√2)
    
    Dot product: 1*2 + 1*2 = 4
    Cosine similarity: 4 / (√2 * 2√2) = 1.0
    
    They point in EXACTLY the same direction! Cosine similarity captures this.
    The dot product (4) doesn't clearly tell us they're the same direction.
    
    Example 2:
    a = [1, 0]
    b = [0, 1]
    
    Dot product: 0
    Cosine similarity: 0
    
    Both tell us they're perpendicular, so in this case they agree!
    
    COMPUTATIONAL NOTE:
    ------------------
    Computing cosine similarity requires:
    1. Computing the dot product: O(n)
    2. Computing both norms: O(n) each
    3. Division: O(1)
    Total: O(n), same as dot product, but with more overhead.
    
    When comparing many vectors pairwise, you can precompute and cache norms
    to avoid recomputing them, making this more efficient.
    
    Example:
    -------
    >>> v1 = Vector([1, 0])
    >>> v2 = Vector([1, 1])
    >>> cosine_similarity(v1, v2)
    0.707...  # cos(45°) ≈ 0.707
    
    >>> v_same_direction = Vector([1, 0])
    >>> v_double = Vector([2, 0])  # Same direction, twice as long
    >>> cosine_similarity(v_same_direction, v_double)
    1.0  # Perfect match in direction!
    """
    dot = v1.dot(v2)
    norm1 = v1.norm()
    norm2 = v2.norm()
    
    if norm1 ==0. or norm2 ==0:
        raise ValueError(
            "Cannot compute cosine similarity with zero vector! "
            "Zero vector has no direction."
        )
    return dot / (norm1 * norm2 )

def projection(v: Vector, onto: Vector) -> Vector:
    """
    Project vector v onto vector onto (the shadow of v on onto).
    
    Parameters:
    ----------
    v : Vector
        The vector to project
    onto : Vector
        The vector to project onto
    
    Returns:
    -------
    Vector : The projection of v onto the line defined by 'onto'
    
    MATHEMATICAL DEFINITION:
    -----------------------
    proj_b(a) = ((a · b) / (b · b)) * b
    
    This gives the vector component of 'a' that points in the direction of 'b'.
    
    GEOMETRIC INTERPRETATION:
    ------------------------
    Imagine shining a light perpendicular to vector 'onto', so that vector v
    casts a shadow onto the line defined by 'onto'. That shadow is the projection!
    
    The projection is always parallel to 'onto' (it lies on the same line).
    It can be shorter, longer, or even opposite to 'onto', but always parallel.
    
    COMPONENTS:
    ----------
    Every vector v can be decomposed into two components relative to another vector b:
    
    1. Parallel component: proj_b(v) - points along b
    2. Perpendicular component: v - proj_b(v) - orthogonal to b
    
    Check: v = proj_b(v) + (v - proj_b(v))
    
    WHY THIS MATTERS:
    ----------------
    Projection answers: "How much of this vector is in that direction?"
    
    MACHINE LEARNING APPLICATIONS:
    -----------------------------
    
    1. GRAM-SCHMIDT ORTHOGONALIZATION:
       To create an orthogonal basis (perpendicular vectors), you repeatedly project
       vectors onto the space spanned by previous vectors and subtract to get the
       perpendicular component. This is used in QR decomposition.
    
    2. PRINCIPAL COMPONENT ANALYSIS (PCA):
       Data is projected onto principal components (directions of maximum variance).
       Projection reduces dimensionality while preserving information.
    
    3. LINEAR REGRESSION:
       The predicted values are the projection of the target vector onto the space
       spanned by the feature columns. The residuals are the perpendicular component.
    
    4. NEAREST NEIGHBOR IN SUBSPACES:
       Find the closest point in a subspace by projecting onto that subspace.
    
    SPECIAL CASES:
    -------------
    
    1. If v is parallel to 'onto': proj(v) = v (entire vector is in that direction)
    
    2. If v is perpendicular to 'onto': proj(v) = 0 (no component in that direction)
    
    3. If 'onto' is a unit vector: proj(v) = (v · onto) * onto
       (Simpler formula because onto · onto = 1)
    
    LENGTH OF PROJECTION:
    --------------------
    ||proj_b(a)|| = |a · b| / ||b||
    
    This is the length of the shadow. The dot product divided by the length of
    the vector you're projecting onto.
    
    Example:
    -------
    >>> v = Vector([3, 4])
    >>> onto = Vector([1, 0])  # Project onto x-axis
    >>> proj = projection(v, onto)
    >>> proj
    Vector([3.0, 0.0])
    
    Interpretation: The x-component of [3, 4] is [3, 0].
    This makes sense: the shadow of [3, 4] on the x-axis is just [3, 0]!
    
    >>> v = Vector([1, 1])
    >>> onto = Vector([1, 0])
    >>> proj = projection(v, onto)
    >>> proj
    Vector([1.0, 0.0])
    
    >>> perp = Vector([v[i] - proj[i] for i in range(len(v))])
    >>> perp
    Vector([0.0, 1.0])
    
    Check: v = proj + perp: [1, 1] = [1, 0] + [0, 1] 
    """
    # proj_b(a) = ((a · b) / (b · b)) * b
    dot_v_onto = v.dot(onto)
    dot_onto_onto = onto.dot(onto)

    if dot_onto_onto == 0:
        raise ValueError(
            "Cannot project onto zero vector! "
            "Zero vector doesn't define a direction."
        )
    #Scale factor 
    scale = dot_v_onto / dot_onto_onto
    
    #Scale the 'onto' vector
    return onto * scale

def vector_rejection(v: Vector, from_: Vector) -> Vector:
    """
    Compute the rejection: the component of v perpendicular to from_.
    
    Parameters:
    ----------
    v : Vector
        The vector to decompose
    from_ : Vector
        The vector to reject from (compute perpendicular component)
    
    Returns:
    -------
    Vector : The component of v that is perpendicular to from_
    
    MATHEMATICAL DEFINITION:
    -----------------------
    rejection = v - projection(v onto from_)
    
    This is the "leftover" part after you remove the parallel component.
    
    GEOMETRIC INTERPRETATION:
    ------------------------
    If projection gives you the shadow, rejection gives you what's left.
    It's the component of v that points perpendicular to from_.
    
    VERIFICATION:
    ------------
    The rejection is orthogonal to from_, meaning their dot product is zero:
    rejection · from_ = 0
    
    Also, v decomposes into parallel and perpendicular components:
    v = projection(v, from_) + rejection(v, from_)
    
    MACHINE LEARNING APPLICATIONS:
    -----------------------------
    
    1. GRAM-SCHMIDT PROCESS:
       Build an orthogonal basis by repeatedly computing rejections.
       Each new basis vector is the rejection of the next vector from all previous ones.
    
    2. RESIDUALS IN REGRESSION:
       The residuals (errors) are the rejection of the target vector from the
       space spanned by the features. They represent the unpredictable part.
    
    3. REMOVING CORRELATIONS:
       If two features are correlated, you can "reject" one from the other to
       get an uncorrelated component. Used in feature decorrelation.
    
    Example:
    -------
    >>> v = Vector([3, 4])
    >>> onto = Vector([1, 0])
    >>> proj = projection(v, onto)
    >>> rej = vector_rejection(v, onto)
    >>> rej
    Vector([0.0, 4.0])
    
    Verification:
    >>> proj.dot(rej)  # Should be 0 (perpendicular)
    0.0
    >>> sum_back = Vector([proj[i] + rej[i] for i in range(len(v))])
    >>> sum_back == v  # Should equal original v
    True
    """
    proj = projection(v, from_)
    # Rejection is v minus its projection
    return v - proj

def gram_schmidt(vectors: List[Vector]) -> List[Vector]:
    """
    Orthogonalize a set of vectors using the Gram-Schmidt process.
    
    Parameters:
    ----------
    vectors : List[Vector]
        A list of linearly independent vectors
    
    Returns:
    -------
    List[Vector] : Orthogonal vectors spanning the same space
    
    MATHEMATICAL DEFINITION:
    -----------------------
    Given vectors v1, v2, ..., vn, produce orthogonal vectors u1, u2, ..., un
    such that:
    1. u_i is orthogonal to all previous u_j (i ≠ j)
    2. {u1, ..., uk} spans the same space as {v1, ..., vk} for each k
    
    ALGORITHM:
    ---------
    1. u1 = v1 (first vector stays as is)
    2. u2 = v2 - proj(v2 onto u1) (remove component parallel to u1)
    3. u3 = v3 - proj(v3 onto u1) - proj(v3 onto u2) (remove components parallel to u1 and u2)
    4. Continue for all vectors
    
    GEOMETRIC INTUITION:
    -------------------
    You're building a coordinate system where all axes are perpendicular!
    
    Start with arbitrary vectors (like arrows pointing in various directions).
    The first arrow becomes your first axis.
    The second arrow: remove its component along the first axis, leaving only the perpendicular part.
    The third arrow: remove its components along the first two axes, leaving only what's new.
    
    You end up with perpendicular axes that span the same space as the original vectors.
    
    WHY ORTHOGONAL BASES ARE USEFUL:
    --------------------------------
    When basis vectors are orthogonal (perpendicular), many computations simplify:
    
    1. Projections become easy: proj(v onto u) = (v · u) * u (no denominator!)
    2. Decompositions are clear: each coefficient is just a dot product
    3. Norms combine simply: ||a + b||² = ||a||² + ||b||² (Pythagorean theorem!)
    
    MACHINE LEARNING APPLICATIONS:
    -----------------------------
    
    1. QR DECOMPOSITION:
       Every matrix A can be decomposed as A = QR where Q has orthonormal columns
       (produced by Gram-Schmidt) and R is upper triangular. Used in solving linear systems.
    
    2. MODIFIED GRAM-SCHMIDT:
       A more numerically stable version used in practice to avoid rounding errors.
    
    3. ORTHOGONAL FEATURE CONSTRUCTION:
       Create uncorrelated features from correlated ones. Each new feature is the
       rejection from all previous features, ensuring independence.
    
    IMPORTANT NOTE:
    --------------
    Input vectors must be LINEARLY INDEPENDENT! If they're not, one vector will be
    in the span of the previous ones, and its rejection will be the zero vector.
    
    Example:
    -------
    >>> v1 = Vector([1, 0, 0])
    >>> v2 = Vector([1, 1, 0])
    >>> v3 = Vector([1, 1, 1])
    >>> orthogonal = gram_schmidt([v1, v2, v3])
    >>> # Check they're orthogonal
    >>> orthogonal[0].dot(orthogonal[1])
    0.0
    >>> orthogonal[0].dot(orthogonal[2])
    0.0
    >>> orthogonal[1].dot(orthogonal[2])
    0.0
    """
    
    if not vectors:
        return []
    
    orthogonal = []
    
    for v in vectors:
        # Start with the original vector
        u = v
        
        # Subtract its projection onto all previous orthogonal vectors
        for prev_u in orthogonal:
            u = u - projection(u, prev_u)
        # Check if u is essentially zero (vectors were dependent)
        if u.norm() < 1e-10:
            raise ValueError(
                "Input vectors are linearly dependent! "
                "Cannot orthogonalize dependent vectors."
            )
        orthogonal.append(u)
    return orthogonal

def orthonormalize(vectors: List[Vector]) -> List[Vector]:
    """
    Create an orthonormal basis: orthogonal AND each vector has length 1.
    
    This is Gram-Schmidt followed by normalization.
    
    Parameters:
    ----------
    vectors : List[Vector]
        Linearly independent vectors
    
    Returns:
    -------
    List[Vector] : Orthonormal vectors (perpendicular and unit length)
    
    WHY ORTHONORMAL IS EVEN BETTER:
    ------------------------------
    Orthonormal means:
    1. Vectors are mutually perpendicular (orthogonal)
    2. Each vector has length exactly 1 (normalized)
    
    With an orthonormal basis, everything becomes simpler:
    - Projecting is just a dot product: proj(v onto u) = (v · u) * u
    - Coordinates are just dot products: coefficient_i = v · u_i
    - The basis matrix has U^T U = I (identity)
    
    MACHINE LEARNING APPLICATIONS:
    -----------------------------
    
    1. PCA:
       Principal components form an orthonormal basis. Each component is unit length
       and perpendicular to all others.
    
    2. SVD (SINGULAR VALUE DECOMPOSITION):
       A = U Σ V^T where U and V have orthonormal columns.
       This is the foundation of many ML algorithms!
    
    3. WHITENING:
       Transform data so features are uncorrelated (orthogonal) and have unit variance
       (normalized). Uses orthonormal basis.
    
    Example:
    -------
    >>> v1 = Vector([1, 0])
    >>> v2 = Vector([1, 1])
    >>> orthonormal = orthonormalize([v1, v2])
    >>> # Check orthogonality
    >>> orthonormal[0].dot(orthonormal[1])
    0.0
    >>> # Check normalization
    >>> orthonormal[0].norm()
    1.0
    >>> orthonormal[1].norm()
    1.0
    """
    # First orthogonalize
    orthogonal = gram_schmidt(vectors)
    
    # Then normalize each vector
    orthonormal = [v.normalize() for v in orthogonal]
    
    return orthonormal

# ==============================================================================
# USAGE EXAMPLES AND TESTS
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DOT PRODUCTS: INTERACTIVE TUTORIAL")
    print("=" * 70)
    
    print("\n### BASIC DOT PRODUCT ###\n")
    
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    
    dot = dot_product(v1, v2)
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v1 · v2 = {dot:.4f}")
    
    print("\n### PERPENDICULAR VECTORS ###\n")
    
    v_right = Vector([1, 0])
    v_up = Vector([0, 1])
    
    dot_perp = dot_product(v_right, v_up)
    print(f"Right: {v_right}")
    print(f"Up: {v_up}")
    print(f"Dot product: {dot_perp:.4f}")
    print("→ They're perpendicular (90 degrees)!")
    
    print("\n### COSINE SIMILARITY ###\n")
    
    doc1 = Vector([2, 1, 0])  # Document: "cat cat dog"
    doc2 = Vector([1, 1, 0])  # Document: "cat dog"
    
    cos_sim = cosine_similarity(doc1, doc2)
    print(f"Doc 1 word counts: {doc1}")
    print(f"Doc 2 word counts: {doc2}")
    print(f"Cosine similarity: {cos_sim:.4f}")
    print("→ High similarity despite different lengths!")
    
    print("\n### PROJECTION ###\n")
    
    v = Vector([3, 4])
    onto_x = Vector([1, 0])
    
    proj = projection(v, onto_x)
    print(f"Vector v: {v}")
    print(f"Project onto x-axis: {onto_x}")
    print(f"Projection: {proj}")
    print("→ The x-component of v is [3, 0]")
    
    print("\n### GRAM-SCHMIDT ORTHOGONALIZATION ###\n")
    
    v1 = Vector([1, 1])
    v2 = Vector([1, 2])
    
    orthogonal = gram_schmidt([v1, v2])
    
    print(f"Original v1: {v1}")
    print(f"Original v2: {v2}")
    print(f"\nOrthogonal u1: {orthogonal[0]}")
    print(f"Orthogonal u2: {orthogonal[1]}")
    
    # Verify they're perpendicular
    dot_check = orthogonal[0].dot(orthogonal[1])
    print(f"\nu1 · u2 = {dot_check:.6f}")
    print("→ Successfully orthogonalized! (dot product ≈ 0)")
    
    print("\n✅ Dot products and projections complete!")
    print("\nNext: matrix_multiplication.py to see how matrices transform space")
