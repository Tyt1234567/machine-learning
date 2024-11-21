import math
import numpy as np
import matplotlib.pyplot as plt

def main():
    """
    Compute your belief in how well someone can see based
    off an eye exam with 20 questions at different fonts
    """

    belief_a = {}
    for i in np.arange(0.01, 1.01, 0.01):
        abi = i
        p = i*0.8+0.1
        belief_a[abi] = p
    normalize(belief_a)

    observations = [{"font_size":1, "is_correct":False},{"font_size":1, "is_correct":False},{"font_size":1, "is_correct":False},{"font_size":1, "is_correct":False},{"font_size":1, "is_correct":False},{"font_size":1, "is_correct":False},{"font_size":1, "is_correct":False},{"font_size":1, "is_correct":False},{"font_size":1, "is_correct":False},{"font_size":1, "is_correct":False}]
    for obs in observations:
        update_belief(belief_a,obs)


def update_belief(belief, obs):
    """
    Take in a prior belief (stored as a dictionary) for a random
    variable representing how well someone can see based on a single
    observation (obs). Update the belief based using Bayes' Theorem
    """
    # loop over every value in the support of the belief RV
    for key,value in belief.items():
        # the prior belief P(A = a)
        prior_a = value
        # the obs probability P(obs | A = a)
        likelihood = calc_likelihood(key, obs)
        # numerator of Bayes' Theorem
        belief[key] = prior_a * likelihood

    # calculate the denominator of Bayes' Theorem
    normalize(belief)

    plt.plot(np.arange(0.01, 1.01, 0.01), belief.values())
    plt.show()
    return belief
def normalize(belief):
    # in place normalization of a belief dictionary
    total = sum(belief.values())
    for key,value in belief.items():
        belief[key] = value/total



def calc_likelihood(a, obs):
    # returns P(obs | A = a) using Item Response Theory
    f = obs["font_size"]
    p_correct_true = sigmoid(a + f - 1)
    if obs["is_correct"]:
        return p_correct_true
    else:

        return 1 - p_correct_true
def sigmoid(x):
    # the classic squashing function
    return 1 / (1 + math.exp(-x))

main()