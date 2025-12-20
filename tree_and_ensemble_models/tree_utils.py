"""
TREE UTILITIES: The Building Blocks for Decision Trees
======================================================

This module contains the core data structures and utilities needed to build
tree-based models. When I started implementing trees, I realized I needed a
flexible Node class that could represent both splits and leaves.

THE NODE CLASS:
--------------
A decision tree is just a bunch of nodes connected together. Each node is either:

1. INTERNAL NODE (Split Node):
   - Asks a question: "Is feature X > threshold?"
   - Has two children: left (answer: yes) and right (answer: no)
   - Stores: feature_index, threshold, left_child, right_child

2. LEAF NODE:
   - Makes a prediction
   - No children
   - Stores: value (the prediction)

This simple structure can represent arbitrarily complex decision boundaries!

DESIGN PHILOSOPHY:
-----------------
I wanted the Node class to be:
- Simple: Easy to understand and debug
- Flexible: Works for both classification and regression
- Inspectable: Can print the tree structure easily
"""

class Node:
    """
    A node in a decision tree.
    
    Can be either:
    - An internal node (has feature_index and threshold for splitting)
    - A leaf node (has value for prediction)
    
    PARAMETERS:
    ----------
    feature_index : int or None
        Which feature to split on (None for leaf nodes)
    
    threshold : float or None
        The split threshold (None for leaf nodes)
    
    left : Node or None
        Left child node (samples where feature <= threshold)
    
    right : Node or None
        Right child node (samples where feature > threshold)
    
    value : float, int, or list, or None
        The prediction value (None for internal nodes)
        - Regression: float (predicted value)
        - Binary classification: int (0 or 1)
        - Multiclass: list of class probabilities
    
    impurity : float or None
        The impurity measure at this node (Gini, Entropy, MSE)
    
    n_samples : int
        Number of training samples that reached this node
    
    STRUCTURE EXAMPLES:
    ------------------
    
    Internal Node:
    Node(feature_index=2, threshold=5.5, left=..., right=...)
    Meaning: "If feature[2] <= 5.5, go left; else go right"
    
    Leaf Node:
    Node(value=1.0)
    Meaning: "Predict 1.0"
    """
    
    def __init__(
        self,
        feature_index=None,
        threshold =None,
        left=None,
        right=None,
        value=None,
        impurity=None,
        n_samples=0
    ):
        # Split properties ( for internal nodes )
        self.feature_index = feature_index
        self.threshold = threshold
        self.right = right
        self.left = left
        
        # Prediction value for leaf node
        self.value = value
        
        # Metadata
        self.impurity = impurity
        self.n_samples = n_samples
        
    def is_leaf(self):
        """
        Check if this node is a leaf (terminal node).
        
        A node is a leaf if it has a prediction value and no children.
        """
        return self.left is None and self.right is None and self.value is not None
    
    def __repr__(self):
        """
        String representation for debugging.
        """
        if self.is_leaf():
            return f"LeafNode(value={self.value}, n_samples={self.n_samples})"
        else:
            return (f"SplitNode(feature={self.feature_index}, "
                   f"threshold={self.threshold:.4f}, n_samples={self.n_samples})")
    

def print_tree(node:Node, depth=0 , prefix="Root: ", feature_names=None):
    """
    Recursively print the tree structure in a readable format.
    
    This function helps visualize what the tree learned by printing it
    in a hierarchical format.
    
    PARAMETERS:
    ----------
    node : Node
        The current node to print
    
    depth : int
        Current depth in the tree (used for indentation)
    
    prefix : str
        Prefix string to show (e.g., "Left: ", "Right: ")
    
    feature_names : list of str or None
        Names of features (if None, use indices)
    
    EXAMPLE OUTPUT:
    --------------
    Root: [100 samples] feature_2 <= 5.5, impurity=0.45
      Left: [60 samples] feature_0 <= 3.2, impurity=0.30
        Left: [40 samples] Predict: 0, impurity=0.10
        Right: [20 samples] Predict: 1, impurity=0.15
      Right: [40 samples] Predict: 1, impurity=0.25
    """
    indent = " " * depth
    
    if node.is_leaf():
        print(f"{indent}{prefix}[{node.n_samples} samples] "
              f"Predict: {node.value}, impurity={node.impurity:.4f}")
    else:
        feature_name = (feature_names[node.feature_index] if feature_names else f"feature_{node.feature_index}") 
        
        print(f"{indent}{prefix}[{node.n_samples} samples] "
              f"{feature_name} <= {node.threshold:.4f}, "
              f"impurity={node.impurity:.4f}")
        if node.left:
            print_tree(node.left, depth + 1, "Left: ", feature_names)
        if node.right:
            print_tree(node.right, depth + 1, "Right: ", feature_names)
            
def get_tree_depth(node:Node):
    """
    Calculate the depth of the tree.
    
    Depth is the length of the longest path from root to leaf.
    
    PARAMETERS:
    ----------
    node : Node
        The root node
    
    RETURNS:
    -------
    int : Maximum depth of the tree
    
    RECURSION LOGIC:
    ---------------
    - Leaf node: depth = 0
    - Internal node: depth = 1 + max(depth of left, depth of right)
    """
    if node is None or node.is_leaf():
        return 0
    
    left_depth = get_tree_depth(node.left) if node.left else 0
    right_depth = get_tree_depth(node.right) if node.right else 0
    
    return 1 + max(left_depth, right_depth)

def count_leaves(node:Node):
    """
    Count the number of leaf nodes in the tree.
    
    PARAMETERS:
    ----------
    node : Node
        The root node
    
    RETURNS:
    -------
    int : Number of leaf nodes
    
    RECURSION LOGIC:
    ---------------
    - Leaf node: count = 1
    - Internal node: count = count(left) + count(right)
    """
    if node is None:
        return 0
    
    if node is node.is_leaf():
        return 1
    
    left_count = count_leaves(node.left) if node.left else 0
    right_count = count_leaves(node.right) if node.right else 0
    
    return left_count + right_count