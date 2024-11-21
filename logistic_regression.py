import numpy as np
import matplotlib.pyplot as plt


# Sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 损失函数 (Log Loss) 修正版本
def compute_loss(y, y_hat, epsilon=1e-10):
    m = y.shape[0]
    # 对预测概率进行修正，避免出现 log(0) 或 log(1)
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    loss = -1 / m * (np.dot(y, np.log(y_hat)) + np.dot((1 - y), np.log(1 - y_hat)))
    return loss


# 梯度下降更新参数
def gradient_descent(X, y, W, b, learning_rate, num_iterations):
    m = X.shape[0]
    losses = []
    for i in range(num_iterations):
        # 计算线性组合 z = XW + b
        z = np.dot(X, W) + b
        # 计算预测值 y_hat = sigmoid(z)
        y_hat = sigmoid(z)

        # 计算损失
        loss = compute_loss(y, y_hat)
        losses.append(loss)

        # 计算梯度
        dw = 1 / m * np.dot(X.T, (y_hat - y))
        db = 1 / m * np.sum(y_hat - y)

        # 更新参数
        W -= learning_rate * dw
        b -= learning_rate * db


        print(f"Iteration {i}, Loss: {loss}, W:{W},b:{b}")
    plt.plot(losses)
    plt.show()

    return W, b


# 预测函数
def predict(X, W, b):
    z = np.dot(X, W) + b
    y_hat = sigmoid(z)
    return np.where(y_hat >= 0.5, 1, 0)


# 示例数据集 (年龄, 胆固醇水平, 是否患病)
X = np.array([[45, 200], [50, 220], [60, 180], [40, 190], [55, 240]])
y = np.array([0, 1, 1, 0, 1])

# 初始化参数 (权重和偏置)
W = np.zeros(X.shape[1])  # W 的维度与特征数量一致
b = 0  # 偏置
learning_rate = 0.008
num_iterations = 10000

# 训练模型
W, b = gradient_descent(X, y, W, b, learning_rate, num_iterations)

# 测试模型
X_test = np.array([[46, 210], [65, 180]])
predictions = predict(X_test, W, b)
print("Predictions:", predictions)
