from scipy import stats
import numpy as np

a = stats.norm(1,1)
b = stats.norm(1.5,1)
c = stats.norm(2,1)
'''
#greedy
class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms  # 臂的数量
        self.epsilon = epsilon  # 探索概率 ε
        self.counts = np.zeros(n_arms)  # 每个臂被选择的次数
        self.values = np.zeros(n_arms)  # 每个臂的估计回报

    def select_arm(self):
        # 以 epsilon 的概率进行探索（随机选择一个臂）
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        # 否则选择估计回报最高的臂
        else:
            return np.argmax(self.values)

    def update(self,chosen_arm,new_reward):
        # 更新选择次数
        self.counts[chosen_arm] += 1
        # 使用增量平均更新该臂的估计回报
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # 更新估计值
        self.values[chosen_arm] = (n-1)/n * value + 1/n * new_reward
        return self.values



n_arms = 3  # 假设有三个臂
epsilon = 0.1  # ε = 0.1，表示10%的时间用于探索
eg = EpsilonGreedy(n_arms, epsilon)


# 模拟1000次选择
for _ in range(10000):
    #每条臂的回报
    true_probs = [a.rvs(size=1), b.rvs(size=1), c.rvs(size=1)]
    chosen_arm = eg.select_arm()
    # 基于真实的成功率模拟一个奖励
    reward = true_probs[chosen_arm]
    total_reward = eg.update(chosen_arm, reward)
    print(f'第{_+1}次选择{chosen_arm+1},更新后的回报为{total_reward}')

'''
#Thompson Sampling
class ThompsonSampling:
    def __init__(self, n_arms):
        # 初始化每个臂的 α 和 β 参数
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self):
        # 对每个臂采样 θ 值
        sampled_theta = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n_arms)]
        print(sampled_theta)
        # 选择拥有最大 θ 值的臂
        return np.argmax(sampled_theta)

    def update(self, chosen_arm, reward):
        # 根据奖励更新该臂的 α 或 β
        if reward == 1:
            self.alpha[chosen_arm] += 1
        else:
            self.beta[chosen_arm] += 1


# 模拟实验
def simulate(thompson_sampler, n_rounds, true_probs):
    n_arms = len(true_probs)
    rewards = np.zeros(n_rounds)
    arm_counts = np.zeros(n_arms)

    for t in range(n_rounds):
        # Thompson Sampling 选择臂
        chosen_arm = thompson_sampler.select_arm()
        print(f'第{t+1}次选择{chosen_arm+1}')
        arm_counts[chosen_arm] += 1
        # 根据真实概率获取奖励
        reward = np.random.binomial(1, true_probs[chosen_arm])
        rewards[t] = reward
        # 更新模型
        thompson_sampler.update(chosen_arm, reward)

    return rewards, arm_counts


# 真实的臂成功概率
true_probs = [0.75, 0.8, 0.85]  # 可以根据需要调整
n_rounds = 1000  # 实验总轮次

# 创建 Thompson Sampling 实例
thompson_sampler = ThompsonSampling(n_arms=len(true_probs))
# 运行模拟
rewards, arm_counts = simulate(thompson_sampler, n_rounds, true_probs)
print(rewards)
print(arm_counts)

#Upper Confidence Bound


