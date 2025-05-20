from cgitb import reset

from perceptron import Perceptron, Activation
from matplotlib import image as img
from matplotlib import pyplot as pp
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
    perceptron = Perceptron(size, 27, [(size, Activation.sigmoid)])  # сама нейросеть
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
        if int(args[0]) and float(args[0]):
            train(int(args[0]),float(args[1]))
        else:
            print("Invalid argument: argument is not of int type")
    elif name == "load":
        load_image(args[0])
    elif name == "help":
        help()
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
def help():
    print("Command list:")
    print("end - complete the process")
    print("put {filename} - put data into the input layer of the neural network")
    print("train {epochs} - train the neural network")
    print("load - load an image to dataset")
def load_dataset():
    for i in range(len(symbols)):
        load_image(f"../res/symbols/{symbols[i]}.png")



def predict(x):
    print("Processing...")
    if perceptron is not None:
        out = perceptron.feedforward(x).astype("float32")[0]
        result = list(out).index(max(list(out)))

        if out[result] < 0.5:
            print("Result: Not an anbur symbol")
        else:
            if result >= 2:
                result+= 1
            pp.imshow(dataset[f"{symbols[result]}.png"].reshape(32,32), cmap="gray")
            pp.show()
            print(f"Result: {symbols[result]} symbol")
    else:
        print("Error: AI is not initialized!")


def train(epochs, learning_rate):
    print("Training neural network...")
    if perceptron is not None:
        analytics = perceptron.train(train_x, train_y, epochs, learning_rate)
        print(analytics)
        analytics.plot_loss()
        analytics.plot_accuracy()
    else:
        print("Error: AI is not initialized!")


run()
print("Please enter the command...")
inp = input()
while not "end" in inp:
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