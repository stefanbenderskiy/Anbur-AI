from perceptron import Perceptron, Activation
from matplotlib import image as img
import numpy as np

size = 32 * 32  # размер одномерной матрицы изображения: ширина * высота
symbols = ["an", "bur", "gai", "doi", "e", "zhoi", "zhoi", "dzhoi", "zata", "dzita", "i", "koke", "lei", "menoe",
           "nenoe", "vooi", "peei", "rei", "sii", "tai", "u", "chery", "shchooi", "yry", "yat", "ver",
           "omega"]  # название символов анбур азбуки
# шаблон для обучения нейросеть
train_x = None
train_y = None
dataset = {}# правильные ответы к шаблону
perceptron = None


def run():
    global perceptron, dataset,train_x, train_y
    print("Creating neural network...")
    perceptron = Perceptron(size, 27, [(2 * size, Activation.sigmoid)])  # сама нейросеть
    print("Loading data...")
    load_dataset()
    train_x = dataset.values()
    train_y = [np.array([[1 if j == i else 0 for j in range(27)]]) for i in range(27)]
    print("Done!")

def command(name,*args):
    global dataset

    if name == "put":
        if args[0] in dataset:
            predict(dataset[args[0]])
        else:
            print("Invalid argument: image name not in dataset.")
    elif name == "train":
        if type(int(args[0])):
            train(int(args[0]))
        else:
            print("Invalid argument: argument is not of int type")
    else:
        return False
commands = ["put","help","load", "train"]

def load_image(filename):
    global dataset
    print(f"loading image:{filename}")
    image = img.imread(filename)
    palette = image.shape[2]
    image = image.reshape(size * palette)
    new_image = []
    p = 0
    for j in range(len(image)):
        p += image[j]
        if (j + 1) % palette == 0:
            new_image.append(p / palette)
            p = 0
    new_image = np.array([new_image])
    dataset[filename.split("/")[-1]] = new_image
def load_dataset():
    for i in symbols:
        load_image(f"../res/symbols/{i}.png")



def predict(x):
    print("Processing...")
    if perceptron is not None:
        out = perceptron.feedforward(x)[0]
        for i in range(len(out)):
            print(f"{symbols[i]} - {round(i * 100, 2)}%")
        result = list(out).index(max(list(out)))
        if out[result] < 0.5:
            print("Result: Not an anbur symbol")
        else:
            print(f"Result: {symbols[result]} symbol")
    else:
        print("Error: AI is not initialized!")


def train(epochs):
    print("Training neural network...")
    if perceptron is not None:
        analytics = perceptron.train(train_x, train_y, epochs, 0.1)
        print(analytics)
        analytics.plot_loss()
        analytics.plot_accuracy()
    else:
        print("Error: AI is not initialized!")


run()
print("Please enter the command...")
inp = input()
while not "/end" in inp:
    s = inp.split()
    flag = False
    for i in range(len(s)):
        if s[i] in commands:
            command(s[i],*s[i + 1:])
            flag = True
            break
    if not flag:
        print("Invalid command")
        print("Please enter the command...")
        flag = False

    inp = input()
print("The process is completed")