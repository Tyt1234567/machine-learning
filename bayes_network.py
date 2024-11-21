import math
import random

def generate_samples(N):
    # P(Uni = 1) = 0.8
    # P(Inf = 1|Uni = 1) = 0.2
    # P(Inf = 1|Uni = 0) = 0.1
    # P(Fev = 1|Inf = 1) = 0.9
    # P(Fev = 1|Inf = 0) = 0.05
    # P(Tir = 1|Uni = 0,Inf = 0) = 0.1
    # P(Tir = 1|Uni = 1,Inf = 0) = 0.8
    # P(Tir = 1|Uni = 0,Inf = 1) = 0.9
    # P(Tir = 1|Uni = 1,Inf = 1) = 1.0

    samples = []
    uni = random.choices([1, 0], weights=[0.8, 0.2], k=N)
    uni_1 = uni.count(1)
    uni_0 = uni.count(0)

    influenced = []
    inf = random.choices([1,0], weights=[0.2,0.8], k=uni_1)
    influenced.extend(inf)
    inf = random.choices([1, 0], weights=[0.1, 0.9], k=uni_0)
    influenced.extend(inf)
    inf_1 = influenced.count(1)
    inf_0 = influenced.count(0)

    fever = []
    fev = random.choices([1, 0], weights=[0.9, 0.1], k=inf_1)
    fever.extend(fev)
    fev = random.choices([1, 0], weights=[0.05, 0.95], k=inf_0)
    fever.extend(fev)

    uni_inf_0_0 = 0
    uni_inf_1_0 = 0
    uni_inf_0_1 = 0
    uni_inf_1_1 = 0
    for i in range(N):
        if influenced[i] == 0 and uni[i] == 0:
            uni_inf_0_0 += 1
        elif influenced[i] == 1 and uni[i] == 0:
            uni_inf_1_0 += 1
        elif influenced[i] == 0 and uni[i] == 1:
            uni_inf_0_1 += 1
        elif influenced[i] == 1 and uni[i] == 1:
            uni_inf_1_1 += 1

    tired = []
    tir = random.choices([1, 0], weights=[0.1, 0.9], k=uni_inf_0_0)
    tired.extend(tir)
    tir = random.choices([1, 0], weights=[0.8, 0.2], k=uni_inf_1_0)
    tired.extend(tir)
    tir = random.choices([1, 0], weights=[0.9, 0.1], k=uni_inf_0_1)
    tired.extend(tir)
    tir = random.choices([1, 0], weights=[1, 0], k=uni_inf_1_1)
    tired.extend(tir)

    for i in range(N):
        samples.append([uni[i], influenced[i], fever[i], tired[i]])

    return samples


def get_any_probability(query,conditions):
    cond = 0
    particles = generate_samples(10000)
    for condition in conditions:
        cond += particles.count(condition)
    return particles.count(query)/cond

print(get_any_probability([1,1,1,1],[[1,1,1,1],[1,1,1,0]]))