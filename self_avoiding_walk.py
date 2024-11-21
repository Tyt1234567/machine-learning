import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)

def rand_walk(num_step, num_iter, moves):
    # 初始化随机游走的统计量
    square_dist = np.zeros(num_iter)
    weights = np.zeros(num_iter)

    for it in range(num_iter):
        trial = 0
        i = 1

        while i != num_step - 1:
            # 初始化
            X, Y = 0, 0
            weight = 1
            lattice = np.zeros((2 * num_step + 1, 2 * num_step + 1))
            lattice[num_step + 1, num_step + 1] = 1
            path = np.array([0, 0])
            xx = num_step + 1 + X
            yy = num_step + 1 + Y

            print("iter: %d, trial %d" % (it, trial))

            for i in range(num_step):
                up = lattice[xx, yy + 1]
                down = lattice[xx, yy - 1]
                left = lattice[xx - 1, yy]
                right = lattice[xx + 1, yy]

                neighbors = np.array([1, 1, 1, 1]) - np.array([up, down, left, right])

                if np.sum(neighbors) == 0:
                    #重新开始
                    i = 1
                    break
                # 更新权重
                weight = weight * np.sum(neighbors)

                direction = np.where(np.random.rand() < np.cumsum(neighbors / float(np.sum(neighbors))))
                #np.cumsum为CDF
                #np.where返回符合条件的索引列表
                X = X + moves[direction[0][0], 0]
                Y = Y + moves[direction[0][0], 1]

                path_new = np.array([X, Y])
                path = np.vstack((path, path_new))

                # 更新网格坐标
                xx = num_step + 1 + X
                yy = num_step + 1 + Y
                lattice[xx, yy] = 1

            trial = trial + 1

        # 计算平方距离
        square_dist[it] = X**2 + Y**2
        weights[it] = weight

    # 计算加权平均平方距离
    mean_square_dist = np.mean(weights * square_dist) / np.mean(weights)
    print("mean square dist: ", mean_square_dist)

    # 生成路径图
    plt.figure()
    for i in range(num_step - 1):
        plt.plot(path[i, 0], path[i, 1], path[i + 1, 0], path[i + 1, 1], 'ob')
    plt.title('Random Walk with No Overlaps')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    # 生成平方距离分布图
    plt.figure()
    sns.displot(square_dist)
    plt.xlim(0, np.max(square_dist))
    plt.title('Square Distance of the Random Walk')
    plt.xlabel('Square Distance (X^2 + Y^2)')
    plt.show()

if __name__ == "__main__":
    num_step = 300  # 步数
    num_iter = 100  # 迭代次数
    moves = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]])  # 可能的移动方向：上、下、左、右
    rand_walk(num_step, num_iter, moves)
