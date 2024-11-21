import numpy as np
import matplotlib.pyplot as plt
import os


class Perceptron:
    def __init__(self, data, label, num_epochs):
        self.num_epochs = num_epochs
        self.data = data
        self.label = label
        self.theta = np.array([0, 0])  # 初始化权重
        self.theta0 = 0  # 初始化偏置

    def fit(self):
        for epoch in range(self.num_epochs):
            for i in range(len(self.data)):
                # 检查误分类
                if self.label[i] * (np.dot(self.theta, self.data[i]) + self.theta0) <= 0:
                    self.theta = self.theta + 0.5 * self.data[i] * self.label[i]
                    self.theta0 = self.theta0 + 0.1 * self.label[i]
            self.plot(epoch, self.theta, self.theta0)
            print(epoch)

    def plot(self, epoch, theta, theta0):
        x = np.array([-3, 3])
        # 计算直线的 y 值，公式为 y = -(theta0 + theta[0] * x) / theta[1]
        y = -(theta[0] * x + theta0) / theta[1]

        # 绘制正确分类的点
        correct_class_0 = self.data[(self.label == -1) & (self.label * (np.dot(self.data, theta) + theta0) > 0)]
        correct_class_1 = self.data[(self.label == 1) & (self.label * (np.dot(self.data, theta) + theta0) > 0)]

        plt.scatter(correct_class_0[:, 0], correct_class_0[:, 1], color='blue', label='Class 0 (Correct)')
        plt.scatter(correct_class_1[:, 0], correct_class_1[:, 1], color='red', label='Class 1 (Correct)')

        # 绘制误分类的点
        misclassified_class_0 = self.data[(self.label == -1) & (self.label * (np.dot(self.data, theta) + theta0) <= 0)]
        misclassified_class_1 = self.data[(self.label == 1) & (self.label * (np.dot(self.data, theta) + theta0) <= 0)]

        plt.scatter(misclassified_class_0[:, 0], misclassified_class_0[:, 1], color='darkblue',
                    label='Class 0 (Misclassified)')
        plt.scatter(misclassified_class_1[:, 0], misclassified_class_1[:, 1], color='darkred',
                    label='Class 1 (Misclassified)')


        # 绘制决策边界
        plt.plot(x, y, color='green')

        plt.title(f"Epoch {epoch + 1}, theta={theta}, theta0={theta0}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()

        # 检查保存路径并创建
        if not os.path.exists('perceptron_results'):
            os.makedirs('perceptron_results')

        plt.savefig(f'perceptron_results/{epoch + 1}.png')
        plt.clf()  # 清空当前图形


if __name__ == "__main__":
    # 生成数据
    np.random.seed(42)  # 固定随机种子以保证结果一致

    # 类别 0
    class_0 = np.random.randn(50, 2) + np.array([-0.5, -0.5])
    labels_0 = np.zeros(50, dtype=int) - np.ones(50, dtype=int)  # 标签 -1
    # 类别 1
    class_1 = np.random.randn(50, 2) + np.array([1.5, 1.5])
    labels_1 = np.ones(50, dtype=int)  # 标签 1

    # 合并数据
    data = np.vstack((class_0, class_1))
    labels = np.hstack((labels_0, labels_1))

    # 打乱数据和标签
    indices = np.random.permutation(len(data))  # 获取打乱后的索引
    data = data[indices]  # 按照打乱的索引重新排列数据
    labels = labels[indices]  # 按照打乱的索引重新排列标签

    # 创建 Perceptron 实例并训练
    Perceptron(data, labels, 100).fit()
