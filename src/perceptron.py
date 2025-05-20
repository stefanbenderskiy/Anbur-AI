import numpy as np
import matplotlib.pyplot as plt


# Функция для генерации двумерной матрицы размером x на y со случайными числами
def generate_matrix(x, y):
    m = []
    for i in range(x * y):
        m.append(np.random.randn())
    return np.array(m).reshape(x, y)


# Класс содержащий функции активации
class Activation:
    def __init__(self, f, df):
        self.f = f
        self.df = df


# Некоторые функции активации
Activation.sigmoid = Activation(lambda x: 1 / (1 + np.exp(-x)),
                                lambda x: np.multiply(1 / (1 + np.exp(-x)), 1 - 1 / (1 + np.exp(-x))))
Activation.tanh = Activation(lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)),
                             lambda x: 1 - np.square((np.exp(x) + np.exp(-x)) / (np.exp(x) + np.exp(-x))))
Activation.relu = Activation(lambda x: np.maximum(0, x), lambda x: 0 * x if np.maximum(0, x) == 0 else 0 * x + 1)
Activation.linear = Activation(lambda x: x, lambda x: x / x)


# Класс слоя нейронов
class Layer:
    def __init__(self, weights, biases, activation=Activation.sigmoid, is_output=False):
        self.weights = weights  # веса для каждого нейрона слоя
        self.biases = biases  # смещения для каждого нейрона слоя
        self.activation = activation  # функция активации нейронов
        self.is_output = is_output  # флаг для обозначаения выходного слоя

    # Функция переднего распространения для слоя
    def forward(self, x):
        return self.activation.f(x.dot(self.weights) + self.biases)

    # Функция корректировки слоя (обратное распространение для слоя)
    def backward(self, x, g, alpha=0.05):

        d = None
        if self.is_output:
            d = g  # в выходном слое значение градиента равно функции потери
        else:
            a = self.activation.f(x.dot(self.weights) + self.biases)  # результат вычисления слоя
            d = np.multiply(g, self.activation.df(a))  # вычисление градиента в точке a

        g = (self.weights.dot((d.transpose()))).transpose()  # вычисление градиента для последующего слоя
        self.weights = self.weights - (alpha * x.transpose().dot(d))
        self.biases = self.biases - (alpha * d)
        return g


class Analytics:
    def __init__(self):
        self.data = []

    def loss(self):
        return [np.sum(np.square(o - y)) / np.size(o) for o, y in self.data]

    def accuracy(self):
        return [1 - i for i in self.loss()]

    def delta(self):
        return [np.sum(o - y) / np.size(o) for o, y in self.data]

    def total_loss(self):
        return self.loss()[-1]

    def total_accuracy(self):
        return self.accuracy()[-1]

    def total_delta(self):
        return self.delta()[-1]

    def average_loss(self):
        return sum(self.loss()) / self.n()

    def average_accuracy(self):
        return sum(self.accuracy()) / self.n()

    def average_delta(self):
        return sum(self.delta()) / self.n()

    def append(self, o, y):
        self.data.append((o, y))

    def n(self):
        return len(self.data)

    def plot_loss(self):
        plt.plot(range(self.n()), self.loss())
        plt.show()

    def plot_accuracy(self):
        plt.plot(range(self.n()), self.accuracy())
        plt.show()

    def plot_delta(self):
        plt.plot(range(self.n()), self.delta())
        plt.show()

    def __repr__(self):
        s = "Analytics:\n"
        for i, d, l, a in zip(range(self.n()), self.delta(), self.loss(), self.accuracy()):
            s = s + f"epoch #{i + 1}:\n delta {round(d * 100, 2)},\n loss {round(l * 100, 2)}%,\n accuracy {round(a * 100, 2)}%\n"
        s = s + f"Total loss:{round(self.total_loss() * 100, 2)}%\n"
        s = s + f"Total accuracy:{round(self.total_accuracy() * 100, 2)}%\n"
        s = s + f"Average loss:{round(self.average_loss() * 100, 2)}%\n"
        s = s + f"Average accuracy:{round(self.average_accuracy() * 100, 2)}%\n"
        s = s + f"Average delta:{round(self.average_delta() * 100, 2)}\n"

        return s

    def __len__(self):
        return self.n()

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, i, o):
        self.data[i] = o
        return self

    def __delitem__(self, i):
        del self.data[i]
        return self

    def __contains__(self, item):
        return item in self.data

    def __iter__(self):
        return iter(self.data)


# Класс нейросети архитектуры Perceptron
class Perceptron:
    def __init__(self, input_size, output_size, layers_params):
        # input_size - размер входного слоя
        # output_size - размер выходного слоя
        # layers_params - параметры слоёв
        self.layers = []
        for i in range(len(layers_params)):  # иницилизация скрытых слоёв
            size, activation = layers_params[i]
            x = 0
            if i == 0:
                x = input_size
            else:
                x = y
            y = size
            self.layers.append(Layer(generate_matrix(x, y), generate_matrix(1, y), activation=activation))
        self.layers.append(
            Layer(generate_matrix(y, output_size), generate_matrix(1, output_size),
                  is_output=True))  # иницилизация выходного слоя

    # Функция переднего распространения (прогнозирования)
    def feedforward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    # Функция обратного распространения (back propagation)
    def backprop(self, x, y, alpha=0.05):
        o = self.feedforward(x)
        g = o - y  # функция потери.
        a = [x]
        for i in range(len(self.layers) - 1):
            x = self.layers[i].forward(x)
            a.append(x)
        for i in range(len(self.layers) - 1, -1, -1):
            g = self.layers[i].backward(a[i], g, alpha)

    # Функция обучения нейросети
    def train(self, train_x, train_y, epochs=1, learning_rate=0.05):
        analytics = Analytics()
        for i in range(epochs):
            outs = []
            for x, y in zip(train_x, train_y):
                self.backprop(x, y, learning_rate)
                outs.append(self.feedforward(x))
            analytics.append(outs, y)
        return analytics
