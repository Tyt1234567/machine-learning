from scipy import stats
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

C14_MEAN_LIFE = 8267
def inference(m = 900):
    """
    Returns a dictionary A, where A[i] contains the
    corresponding probability, P(A = i| M = m).
    m is the number of C14 molecules remaining and i
    is age in years. i is in the range 100 to 10000
    """
    A = {}
    for i in tqdm(range(100,10000+1)):
        A[i] = calc_likelihood(m, i) # P(M = m | A = i)
        # implicitly computes the normalization constant
        normalize(A)

    x = list(A.keys())
    y = list(A.values())
    print(sum(y))
    plt.plot(x,y)
    plt.show()
    return A
def calc_likelihood(m, age):
    """
    Computes P(M = m | A = age), the probability of
    having m molecules left given the sample is age
    years old. Uses the exponential decay of C14
    """
    n_original = 1000
    n_decayed = n_original - m
    p_single = 1 - math.exp(-age/C14_MEAN_LIFE)
    return stats.binom.pmf(n_decayed, n_original, p_single)
def normalize(prob_dict):
    # first compute the sum of the probability
    sum = 0
    for key, pr in prob_dict.items():
        sum += pr
    # then divide each probability by that sum
    for key, pr in prob_dict.items():
        prob_dict[key] = pr / sum
    # now the probabilities sum to 1 (aka are normalized)

print(inference())