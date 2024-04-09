import numpy as np

class SimpleLinear:
    def __init__(self, input_dim, output_dim):
        # 初始化权重和偏置
        self.weights = np.random.rand(input_dim, output_dim)
        self.bias = np.random.rand(output_dim)
    
    def forward(self, x):
        # 执行线性变换
        y = np.dot(x, self.weights) + self.bias
        return y

class ManualLinearRegression:
    def __init__(self, learning_rate=0.01):
        self.weights = np.random.rand(1)
        self.bias = np.random.rand(1)
        self.learning_rate = learning_rate
    
    def forward(self, x):
        return x * self.weights + self.bias
    
    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, x, y_pred, y_true):
        # 计算损失函数对预测值的导数
        d_loss_y_pred = 2 * (y_pred - y_true) / y_true.size
        
        # 计算预测值对权重的导数
        d_y_pred_d_weights = x
        
        # 计算预测值对偏置的导数，即1
        d_y_pred_d_bias = 1
        
        # 通过链式法则计算损失对权重的导数
        d_loss_d_weights = np.dot(d_y_pred_d_weights.T, d_loss_y_pred)
        
        # 通过链式法则计算损失对偏置的导数
        d_loss_d_bias = np.sum(d_loss_y_pred)
        
        return d_loss_d_weights, d_loss_d_bias
    
    def update_weights(self, d_weights, d_bias):
        self.weights -= self.learning_rate * d_weights
        self.bias -= self.learning_rate * d_bias
    
# 示例数据
x = np.random.rand(100, 1)
y_true = 3 * x + 2 + np.random.rand(100, 1) * 0.5

# 模型实例化和训练
model = ManualLinearRegression(learning_rate=0.1)
for epoch in range(100):
    y_pred = model.forward(x)
    loss = model.compute_loss(y_pred, y_true)
    d_weights, d_bias = model.backward(x, y_pred, y_true)
    model.update_weights(d_weights, d_bias)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

print(f"Trained weights: {model.weights}, Trained bias: {model.bias}")
