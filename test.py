import load_data
from neural_network import *
train_data = load_data.loadMnist("./unzipped_data")
test_data = load_data.loadMnist("./unzipped_data", type="test")
network = NeuralNetwork([784, 30, 10])
network.train(30, 3.0, 20, train_data)
