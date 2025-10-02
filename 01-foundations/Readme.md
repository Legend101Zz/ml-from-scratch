# Day1: The Foundations

> “Tell me and I forget, teach me and I may remember, involve me and I learn.” — Benjamin Franklin

## Our Mission: Building Mathematical Intuition

Before diving into the exciting world of algorithms that can recognize faces or predict stock prices, I'm starting with a fundamental truth: every sophisticated algorithm is built from simple mathematical operations combined in clever ways. When you call `model.fit()` in a library, the computer is just performing millions of additions, multiplications, and comparisons.

My goal here is to build "mathematical intuition." I want to understand not just _what_ an algorithm does, but _why_ it works, when it might fail, and how it could be modified. To do that, I'm forcing myself to implement every mathematical operation from scratch.

For now, optimization is not the goal; understanding is. When we deal with larger datasets later, I'll gladly stand on the shoulders of giants and use the masterpieces that are NumPy and Pandas. But for this initial leg of the journey, we write the code ourselves.

(The one exception is visualization. I'm sane enough to know that implementing a plotting library is a whole other adventure, so we'll be using `matplotlib` to see our results!)

---

## Why This Path Matters

Machine learning algorithms are fundamentally mathematical procedures that find patterns by optimizing functions.

Consider predicting house prices. A simple linear regression finds the "best" line through the data points. Mathematically, "best" means minimizing the error between predicted and actual prices. This single task requires matrix operations, derivatives, and statistical calculations.

Every operation we build here will be a recurring character in our story.

- **Matrix multiplication** is the engine of neural networks.
- **Mean and variance** are the lenses through which we understand our data.
- **Dot products** are the rulers we use to measure similarity.

By implementing these myself, I'm learning the vocabulary of machine learning at its deepest level.

## Our First Mathematical Implementations

Let us begin with the most fundamental operation in machine learning: the dot product. This operation appears everywhere in machine learning, from simple linear regression to complex deep learning networks. If you'll see the file `math_operations.py` you'll find the fn `dot_product` ...

You can go through the function and even read the doc strings to understand what is does , to summarise :

The dot product might seem simple, but it encodes a profound mathematical concept: it measures how much two vectors align with each other. In machine learning, this alignment measurement appears constantly. When a neural network makes a prediction, it is computing dot products between input data and learned parameters. When we measure similarity between documents in text analysis, we often use dot products of word frequency vectors.

But then the question arises :

How does just multiplying two vectors in each dimension measure similarity between them ?

images here :

Next,we'll implement matrix multiplication, which extends the dot product concept to higher dimensions:

again in `math_operations.py` you'll find the fn `matrix_multiply` ...
there's nothing much to add for this so yeah maybe. some hand on practice to get a sense of what we are doing ... just maybe I'll answer why we do matrix multiplication as we do it and how is it related to vectors

iamges here ...

## Statistical Foundations for Data Understanding

Machine learning algorithms make sense of data by understanding its statistical properties. Before an algorithm can find patterns, it needs to understand what the data looks like on average, how much it varies, and how different features relate to each other.

you can go through the `statistics.py` to see the fundamental functions I defined ...
