import pandas as pd
import numpy as np

# 更新参数，训练模型
def train(data_train, label_train, epoch):
    num = data_train.shape[0] #行数
    dim = data_train.shape[1] #列数
    bias = 0                  #偏置值初始化
    weights = np.ones(dim)    #权重初始化为一个57维的权重数组
    learning_rate = 1  # 初始学习率
    bg2_sum = 0       # 用于存放偏置值的梯度平方和
    wg2_sum = np.zeros(dim)  # 用于存放权重的梯度平方和

    for i in range(epoch):
        b_g = 0 #初始化 bias的梯度
        w_g = np.zeros(dim) #初始化 weight为0的梯度
        # 在所有数据上计算梯度，梯度计算时针对损失函数求导
        for j in range(num): # 对每一行求y_pre 及偏置
            y_pre = weights.dot(data_train[j, :]) + bias #求出每一行数据的y
            y_sig = 1 / (1 + np.exp(-y_pre)) # sigmoid 函数压缩结果为0-1概率值
            b_g += (-1) * (label_train[j] - y_sig) #计算bias 梯度偏置
            for k in range(dim): # 计算每一个w 的梯度偏置
                w_g[k] += (-1) * (label_train[j] - y_sig) * data_train[j, k]    #损失函数正则化
                #w_g[k] += (-1) * (label_train[j] - y_sig) * data_train[j, k]+ 0.0002 * weights[k] #损失函数正则化
        #偏置求平均
        b_g /= num
        w_g /= num

        # adagrad 公式更新权重和偏置
        # 先求偏置的平方和
        bg2_sum += b_g ** 2
        wg2_sum += w_g ** 2
        bias = bias -  learning_rate / bg2_sum ** 0.5 * b_g
        weights = weights - learning_rate / wg2_sum ** 0.5 * w_g

        # 每训练100轮，输出一次在训练集上的正确率
        # 在计算loss时，由于涉及到log()运算，因此可能出现无穷大，计算并打印出来的loss为nan
        # 有兴趣的同学可以把下面涉及到loss运算的注释去掉，观察一波打印出的loss
        if (i+1) % 10 == 0:
            # loss = 0
            acc = 0
            result = np.zeros(num)
            for j in range(num):
                y_pre = weights.dot(data_train[j, :]) + bias
                sig = 1 / (1 + np.exp(-y_pre))
                if sig >= 0.5:
                    result[j] = 1
                else:
                    result[j] = 0

                if result[j] == label_train[j]:
                    acc += 1.0
                # loss += (-1) * (y_train[j] * np.log(sig) + (1 - y_train[j]) * np.log(1 - sig))
            # print('after {} epochs, the loss on train data is:'.format(i), loss / num)
            print('after {} epochs, the acc on train data is:'.format(i+1), acc / num)
    return weights, bias

# 验证模型效果
def validate(x_val, y_val, weights, bias):
    num = x_val.shape[0]
    # loss = 0
    acc = 0
    result = np.zeros(num)
    for j in range(num):
        y_pre = weights.dot(x_val[j, :]) + bias
        sig = 1 / (1 + np.exp(-y_pre))
        if sig >= 0.5:
            result[j] = 1
        else:
            result[j] = 0

        if result[j] == y_val[j]:
            acc += 1.0
        # loss += (-1) * (y_val[j] * np.log(sig) + (1 - y_val[j]) * np.log(1 - sig))
    return acc / num

def main():
    df = pd.read_csv('spam_train.csv')
    df = df.fillna(0) # 空值填0
    # (4000, 59)
    array = np.array(df)
    # (4000, 1:58)
    x = array[:, 1:-1]  #x为第二列 到 倒数第二列
    #逗号之前为要取的num行下标范围，逗号之后为要取的num列下标范围；
    # scale 可以将这两列分别除上每列的平均值，把数值范围拉到1附近
    x[:, -1] /= np.mean(x[:, -1])
    x[:, -2] /= np.mean(x[:, -2])
    # (4000, )
    y = array[:, -1]  #y为最后一列的元素,即Label

    # 划分训练集与验证集
    data_train, data_val = x[0:3500, :], x[3500:4000, :]
    label_train, label_val = y[0:3500], y[3500:4000]

    epoch = 120  # 训练轮数
    w, b = train(data_train, label_train, epoch)
    acc = validate(data_val, label_val, w, b)
    print('The acc on val data is:', acc)

if __name__ == '__main__':
    main()