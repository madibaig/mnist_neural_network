from mnist import MNIST
import numpy as np

def loadMnist(path, type="train"):
    #File path of data
    mndata = MNIST(path)

    #Change type of what kind of data to get
    if(type=="train"):
        images, labels = mndata.load_training()
    elif(type=="test"):
        images, labels = mndata.load_testing()

    #Format data to fit numpy and neural network
    formattedImages = mndata.process_images_to_numpy(images)
    #images are reshaped to fit the data structure required for the neural network
    formattedImages = [map_brightness(image).reshape((784, 1)) for image in formattedImages]
    formattedLabels = [vectorized_result(label) for label in labels]

    print("Formatted image [0]", formattedImages[0])
    #print(formattedLabels[0])

    #Returns list of tuples containing (x, y)
    return list(zip(formattedImages, formattedLabels))

def vectorized_result(j):
    """[TAKEN FROM https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_loader.py]
    Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def map_brightness(brightness):
    return (brightness / 256) - 0.5

