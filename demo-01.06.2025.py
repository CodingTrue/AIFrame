"""
DEMO CODE FOR AIFRAME FRAMEWORK
01.06.2025
Written by CodingTrue
"""

###########
# IMPORTS #
###########
from aiframe import NeuralNetwork
from aiframe.Criterion import CrossEntropyCriterion
from aiframe.program import ProgramBuilder
from aiframe.node.Nodes import HiddenLayerNode, ReluActivationNode, SoftmaxActivationNode

from time import perf_counter

import numpy as np
import importlib.util as imputil

import shutil
import os
import gzip
import requests

if imputil.find_spec("gzip") is None:
    print("Please install 'gzip' for this demo to work properly.")
    exit()

if imputil.find_spec("requests") is None:
    print("Please install 'requests' for this demo to work properly.")
    exit()

#############################
# DOWNLOADING MNIST DATASET #
#############################
def download_file(url, path, name):
    print(f"Downloading: {url}")

    stream = requests.get(url=url, stream=True)
    stream.raise_for_status()

    tmp_gz_path = path + "tmp/" + name
    os.makedirs(path + "tmp/", exist_ok=True)

    with open(tmp_gz_path, "wb") as f:
        shutil.copyfileobj(stream.raw, f)

    with gzip.open(tmp_gz_path, "rb") as gz_file:
        with open(path + name, "wb") as file:
            shutil.copyfileobj(gz_file, file)
    os.remove(tmp_gz_path)

base_dir = "data/"
urls = {
    "train_images_60k": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels_60k": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images_10k": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "test_labels_10k": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
}

for name, url in urls.items():
    if not os.path.exists(base_dir + name):
        download_file(url=url, path=base_dir, name=name)
if os.path.exists(base_dir + "tmp"): os.removedirs(base_dir + "tmp")

############################
# EXTRACTING MNIST DATASET #
############################
def read_mnist_images(count: int = 0, path=""):
    bytes_to_read = 16 + count * 784
    with open(path, "rb") as f:
        result = (np.frombuffer(f.read(bytes_to_read)[16:], dtype=np.ubyte) / 255).reshape(count, 784)
    return result

def read_mnist_labels(count: int = 0, path = ""):
    bytes_to_read = 8 + count
    with open(path, "rb") as f:
        b = f.read(bytes_to_read)[8:]
        result = np.frombuffer(b, dtype=np.ubyte)
    return result

print(f"Extracting MNIST data...")

train_samples = 60000
mini_batch_size = 100
mini_batch_count = int(train_samples // mini_batch_size)

train_images = read_mnist_images(count=train_samples, path=base_dir + "train_images_60k").reshape(mini_batch_count, mini_batch_size, 784)
train_labels = read_mnist_labels(count=train_samples, path=base_dir + "train_labels_60k").reshape(mini_batch_count, mini_batch_size)

one_hot_vector = np.eye(10)[train_labels]
training_data = [train_images, one_hot_vector]

########################
# NEURAL NETWORK SETUP #
########################

print("Setup of Neural Network and cross-entropy trainer...")

nn = NeuralNetwork(network_nodes=[
    HiddenLayerNode(100),
    ReluActivationNode(),
    HiddenLayerNode(10),
    SoftmaxActivationNode()
], max_input_size=784).allocate_input_dimensions(random=True)

#################
# TRAINER SETUP #
#################
trainer = ProgramBuilder.create_train_program(nn=nn, criterion=CrossEntropyCriterion())
trainer.assamble()

############
# TRAINING #
############
print("-----------------------")
print("Starting training...")

learn_rate = 0.1
learn_rate_decay = 0.98
iterations = 25

print(f"learn_rate: {learn_rate} | learn_rate_decay: {learn_rate_decay} | epochs: {iterations}")

start = perf_counter()
trainer.train(nn=nn, training_data=training_data, learn_rate=learn_rate, iterations=iterations, learn_rate_decay=learn_rate_decay)
end = perf_counter()

print("Training finished...")
print(f"Training took {end - start:0.2f} seconds.")
print("-----------------------")
print("Evaluating test data...")
print("-----------------------")

############
# EVALUATE #
############
tests = 10000
test_images = read_mnist_images(count=tests, path=base_dir + "test_images_10k")
test_labels = read_mnist_labels(count=tests, path=base_dir + "test_labels_10k")

rights = 0
for i in range(tests):
    output = nn.forward(input_data=test_images[i])
    index = output.argmax()
    real = test_labels[i]

    if real == index: rights += 1

print(f"{rights} / {tests} -> ({rights / tests:0.2f} %)")