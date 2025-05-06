import numpy as np

def generate_matrix(x, y):
    m = []
    for i in range(x * y):
        m.append(np.random.randn())
    return np.array(m).reshape(x, y)
class Activation:
    def __init__(self, f, df):
        self.f = f
        self.df = df

Activation.sigmoid = Activation(lambda x: 1 / (1 + np.exp(-x)),
                                lambda x: np.multiply(1 / (1 + np.exp(-x)), 1 - 1 / (1 + np.exp(-x))))
Activation.tanh = Activation(lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)),
                                lambda x: 1 - np.square((np.exp(x) + np.exp(-x)) / (np.exp(x) + np.exp(-x))))
class Layer: # Класс слоя нейронов
    def __init__(self, weights, biases, activation=Activation.sigmoid, is_output=False):
        self.weights = weights  # веса для каждого нейрона слоя
        self.biases = biases  # смещения для каждого нейрона слоя
        self.activation = activation # функция активации нейронов
        self.is_output = is_output
    def forward(self, x): # переднее распрострнение
        return self.activation.f(x.dot(self.weights) + self.biases)

    def backward(self, x, g, alpha = 0.05):#функция обратного распростронения для слоя
        z,d = None, None
        if self.is_output:
            d = g
        else:
            z = x.dot(self.weights) + self.biases #суммирование
            d = np.multiply(g,self.activation.df(z))#вычисление градиента для
        g = (self.weights.dot((d.transpose()))).transpose() #вычисление градиента для последующего слоя
        self.weights = self.weights - (alpha * x.transpose().dot(g))
        self.biases = self.biases - (alpha * g)
        return g
class Analytics:
    def __init__(self):
        self.data = []
    def loss(self):
        return [np.sum(np.square(o - y)) / np.size(o) for o,y in self.data]
    def accuracy(self):
        return 1 - self.loss()
    def delta(self):
        return [np.sum(o - y) / np.size(o) for o,y in self.data]
    def total_loss(self):
        return self.loss()[-1]
    def total_accuracy(self):
        return self.accuracy()[-1]
    def total_delta(self):
        return self.delta()[-1]
    def average_loss(self):
        return sum(self.loss())/self.n()
    def average_accuracy(self):
        return sum(self.accuracy())/self.n()
    def append(self,o,y):
        self.data.append((o,y))
    def n(self):
        return len(self.data)
class Perceptron: # Класс нейросети

    def __init__(self, input_size, output_size, layers_params):
        self.layers = []
        for i in range(len(layers_params)):
            size, activation = layers_params[i]
            x = 0
            if i == 0:
                x = input_size

            else:
                x = y
            y = size
            self.layers.append(Layer(generate_matrix(x,y),generate_matrix(1,y),is_output= i==0))
        self.layers.append(Layer(generate_matrix(y, output_size),generate_matrix(1,output_size)))
    def feedforward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def backprop(self,x,y, alpha = 0.05):
        o = self.feedforward(x)
        g = y - o
        a = []
        for i in range(len(self.layers) - 1):
            x = self.layers[i].forward(x)
            a.append(x)
        for i in range(len(self.layers) -1, -1, -1):
            g = self.layers[i].backward(a[i],g,alpha)
    def train(self,train_x,train_y, epochs = 1,learning_rate=0.05):
        analytics = Analytics()
        for i in range(epochs):
            for x,y in zip(train_x,train_y):
                self.backprop(x,y,learning_rate)
                o = self.feedforward(x)
                analytics.append(o,y)
        return analytics
