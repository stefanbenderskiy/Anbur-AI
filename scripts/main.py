from perceptron import Perceptron,Activation
from matplotlib import image as img
import numpy as np

size = 32 * 32 * 4  # размер матрицы: ширина * высота * RGBA
symbols = []
train_x = [img.imread(f"../res/smiles/{i}.png").reshape(1, size) for i in symbols]
train_y = [np.array([[1 if j == i else 0 for j in range(15)]]) for i in range(15)]
perceptron = Perceptron(size, 15,[(size, Activation.sigmoid)])

def predict(x):
    out = perceptron.feedforward(x)
    mx = -100000000
    index = -1
    for i in range(out[0].size):
        if out[0][i] > mx:
            index = i
            mx = out[0][i]
    if mx >= 0.5:
        return f"{symbols[index]} ({round(mx * 100, 2)}%)"
    else:
        return f"Not an anbur symbol ({round((1 - mx) * 100,2)}%)"
def train(epochs):
    analytics = perceptron.train(train_x, train_y, epochs, 0.05)
    return analytics