import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from scipy.stats import norm

# Target Distribution: double gamma distribution
dg = stats.dgamma(a=1)  # 定义双伽马分布
qx = stats.norm(loc=0, scale=2)

# Generate samples for PDF
x = np.linspace(min(dg.ppf(0.001), norm.ppf(0.001)), max(dg.ppf(0.999), norm.ppf(0.999)), 1000)
dg_samples = dg.pdf(x)
norm_samples = qx.pdf(x)

# Find scaling constant K for envelope
K = max(dg_samples / norm_samples)

# Plot
df = pd.DataFrame({'Target': dg_samples, 'Envelope': K * norm_samples, 'primitive':norm_samples}, index=x)
ax = df.plot(style=['--', '-',':'], color=['black', 'blue','green'], figsize=(8,6), linewidth=2.0)
ax.plot((2, 2), (0, dg.pdf(2)), 'g--', linewidth=2.0)
ax.plot((2, 2), (dg.pdf(2), K * qx.pdf(2)), 'r--', linewidth=2.0)

ax.text(1.0, 0.20, 'Reject')
ax.text(1.0, 0.03, 'Accept')

def rejection_sampling():
    while True:
        # Re-use global parameters from above
        x = np.random.normal(0, 2) #正态分布中生成一个随机数
        envelope = K * qx.pdf(x)
        p = np.random.uniform(0, envelope)
        if p < dg.pdf(x):
            return x

# Generation samples from rejection sampling algorithm
samples = [rejection_sampling() for x in range(10000)]

# Plot Histogram vs. Target PDF
df['Target'].plot(color='blue', style='--', figsize=(8,6), linewidth=2.0)
pd.Series(samples).hist(bins=300, density=True, color='green', alpha=0.3, linewidth=0.0)
plt.legend(['Target PDF', 'Rejection Sampling'])
plt.show()
