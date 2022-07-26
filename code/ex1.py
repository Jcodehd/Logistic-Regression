from turtle import pos
from cv2 import multiply
from numpy import *;
from pandas import *;
from plotData import plotData;
from costFunction import costFunction;
from gradientDescent import gradientDescent;
from sigmoid import sigmoid;
from predict import predict;


# 读取数据 前两列为成绩，最后一列为分类信息
data = mat(read_table('Machine Learning\Logistic Regression\ex2data1.txt', sep=',', header=None));

X = data[:,:2];
y = data[:,2:];

# 样本数量
m = X.shape[0];
# 特征
n = X.shape[1];

# 画出数据散点图
plotData(X, y);

# 添加一列X0
X = hstack((ones((m,1), dtype=int), X));

# 初始化参数theta
theta = zeros((n+1,1));

# 求出初始损失
cost, grad = costFunction(X, y, theta);

# 设置学习率、迭代次数
alpha = 0.00104;
iters = 200000;

# 梯度下降
cost_, theta_ = gradientDescent(X, y, theta, alpha, iters);
print(theta_)

# 对成绩为45、85的学生进行预测

predict_ = sigmoid(dot(mat([1,45,85]), theta_));

accuracy = predict(X, y, theta_)

print('当学生两科成绩为45、85时，录取概率为: ',predict_);

print('训练模型的精度为: ', accuracy);






