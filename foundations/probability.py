import math


def gaussian_probability(x, mean , std_dev):
    """ 
    Calculate the probability density of a value in a Gaussian (normal) distribution.
    
    The Gaussian distribution is the most important probability distribution in statistics and machine Learning . Many natural phenomea follow this bell-shaped curve, and many ML algorihtms assume data is normally distributed...
    
    The formula: (1 / (σ√(2π))) * e^(-((x-μ)²)/(2σ²))
    where μ is the mean and σ is the standard deviation
    
    Args:
    x (float): The value to calculate probability for
    mean (float): Mean (μ) of the distribution
    std_dev (float): Standard deviation (σ) of the distribution
    """
    if std_dev <=0:
        raise ValueError("Standard deviation must be positive")
    
    # Calculate the exponent : -((x-μ)²)/(2σ²)
    exponent = -((x-mean)**2) / (2*(std_dev)**2)
    
    # Calculate the coefficient : 1 / (σ√(2π))
    coefficient = 1 / (std_dev * math.sqrt(2 * math.pi))
    
    probability_density = coefficient * math.exp(exponent)
    return probability_density

def calculate_probability(favourable_outcome, total_outcomes):
    """ 
    Calculate basic probability as favourable outcomes divided by total outcomes. ( Note this is the Naive definition of probability , read-more: https://medium.com/statistics-theory/what-is-naive-probability-db4d7fb1e785 )
    
    In ML we often estimate probabilities by counting occurences in our training data .
    
        Args:
        favorable_outcomes (int): Number of times the event occurred
        total_outcomes (int): Total number of observations
        
    Returns:
        float: Probability between 0 and 1
        
    Example in ML:
        If 60 out of 100 emails in training data are spam, the probability
        of an email being spam is 60/100 = 0.6
    """
    if total_outcomes <= 0 :
        raise ValueError("Total outcomes must be positive")
    if favourable_outcome < 0 or favourable_outcome > total_outcomes:
        raise ValueError("Favourable outcomes must be between 0 and total outcomes")
    
def conditional_probabiity(prob_a_and_b , prob_b):
    """ 
    Calculate P(A|B) - the probability of A given that B has occurred .
    
    Conditional probability is crucial for understanding how ML models update their beliefs based on evidence.
    
    Formula: P(A|B) = P(A and B) / P(B)
    
    Args:
        prob_a_and_b (float): Probability of both A and B occurring
        prob_b (float): Probability of B occurring
        
    Returns:
        float: Conditional probability P(A|B)
    """ 
    if prob_b == 0 :
        raise ValueError("Cannot condition on impossible event P(B) = 0")
    return prob_a_and_b/prob_b

def bayes_theorem(prior , likelihood, evidence):
    """ 
    Apply Bayes' theorem to update belief based on evidence
    
    Bayes' theorem is one of the most important formulas in ML .
    It tells us how to update our beliefs (prior probability) when we observe new evidence.
    
    Formula: P(hypothesis|evidence) = P(evidence|hypothesis) * P(hypothesis) / P(evidence)
    
    Args:
        prior (float): P(hypothesis) - our initial belief before seeing evidence
        likelihood (float): P(evidence|hypothesis) - how likely the evidence is if hypothesis is true
        evidence (float): P(evidence) - overall probability of seeing this evidence
        
    Returns:
        float: Posterior probability - updated belief after seeing evidence
        
    Intuitive Explanation:
        Imagine you want to know if someone has a disease (hypothesis) given a
        positive test result (evidence):
        - prior: How common is the disease? (say 1%)
        - likelihood: If someone has disease, how likely is positive test? (say 90%)
        - evidence: Overall, how often are tests positive? (say 10%)
        - posterior: Given positive test, what's probability of disease?
          = 0.90 * 0.01 / 0.10 = 0.09 or 9%
    """
    if evidence == 0:
        raise ValueError("Evidence probabilty cannot be zero")
    
    posterior = (likelihood * prior) / evidence
    return posterior