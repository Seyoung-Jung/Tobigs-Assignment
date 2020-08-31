import numpy as np

#시그모이드 함수
def sigmoid(x):
    return 1 / (1+np.exp(-x))

#편미분 함수
def numerical_derivative(f, x):
    delta_x = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index        
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x) # f(x+delta_x)

        x[idx] = tmp_val - delta_x 
        fx2 = f(x) # f(x-delta_x)
        grad[idx] = (fx1 - fx2) / (2*delta_x)

        x[idx] = tmp_val 
        it.iternext()   

    return grad

#========================================================================================#

class LogisticRegression_cls:
    def __init__(self, X_train, y_train, W=np.random.rand(1,1), b=np.random.rand(1), learning_rate=1e-2):
        self.X_train = X_train
        self.y_train = y_train
        self.W = W
        self.b = b
        self.learning_rate = learning_rate
        
    #손실함수
    def loss_func(self):
#         delta = 1e-7
        z = np.dot(self.X_train, self.W) + self.b
        y = sigmoid(z)
        return -np.sum(self.y_train*np.log(y+1e-7) + (1-self.y_train)*np.log((1-y)+1e-7))

        
    #손실 값 계산 함수
    def error_val(self):
        return self.loss_func()
    
             
    #예측 함수
    def predict(self, X):
        result=[]
        for x in X:
            z=np.dot(x, self.W) + self.b
            y=sigmoid(z)

            if y > 0.5:
                result.append(1)
            else:
                result.append(0)

        return result
    
    #학습 함수
    def train(self):
        f = lambda x : self.loss_func()
        print("Initial error value = ", self.error_val())

        for step in range(10001):
            self.W -= self.learning_rate * numerical_derivative(f, self.W)
            self.b -= self.learning_rate * numerical_derivative(f, self.b)
            if (step % 400 == 0):
                print("step = ", step, "error value = ", self.error_val())               