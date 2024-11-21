import numpy as np
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt


bhutan_sample = np.random.normal(loc=7, scale=1, size=200)  # 不丹的幸福感评分
nepal_sample = np.random.normal(loc=6.8, scale=1, size=300)   # 尼泊尔的幸福感评分

def pvalue_bootstrap(bhutan_sample,nepal_sample):
    N = len(bhutan_sample)
    M = len(nepal_sample)
    universal_sample = np.concatenate([bhutan_sample, nepal_sample])
    count = 0
    observed_difference = np.mean(bhutan_sample) - np.mean(nepal_sample)
    differences = []
    for _ in tqdm(range(100000)):
        # 从合并的样本中重新抽取 N 和 M 个样本，分别作为新的不丹和尼泊尔样本
        bhutan_resample = np.random.choice(universal_sample, size=N, replace=True)
        nepal_resample = np.random.choice(universal_sample, size=M, replace=True)
        # 计算新的样本均值
        mu_bhutan = np.mean(bhutan_resample)
        mu_nepal = np.mean(nepal_resample)
        # 计算两者的均值差
        mean_difference = mu_nepal - mu_bhutan
        differences.append(mean_difference)
        # 如果重新抽样的均值差大于原始观察到的差异，计数器加1
        if mean_difference > observed_difference:
            count += 1
    print(np.quantile(differences, 0.025))
    plt.hist(differences, bins=500, density=True, alpha=0.6, color='g')

    plt.show()
    return count/100000
print(pvalue_bootstrap(bhutan_sample,nepal_sample))