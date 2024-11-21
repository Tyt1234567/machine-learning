import matplotlib.pyplot as plt
import math
import numpy as np

def gamma(x, a, b):
    results = []
    for i in x:
        # 使用 math.gamma 计算 Gamma 函数
        coefficient = (b ** a) / math.gamma(a)
        result = coefficient * (i ** (a - 1)) * math.exp(-b * i)
        results.append(result)
    return results

x = np.linspace(0,20,1000)
result_x = gamma(x,2,0.1)
result_y = gamma(x,2,0.2)
result_z = gamma(x,2,0.3)

plt.plot(x,result_x,c='r')
plt.plot(x,result_y,c='g')
plt.plot(x,result_z,c='b')
plt.show()