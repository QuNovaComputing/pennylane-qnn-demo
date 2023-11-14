# pytorch dataset loading, model definition, and model training code
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# pennylane imports for defining the quantum part of the model
import pennylane as qml
import pennylane.numpy as np


# for timing the training process
import time

# load the MNIST training and testing datasets
training_data = datasets.MNIST(
    root="~/pytorch_data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="~/pytorch_data",
    train=False,
    download=True,
    transform=ToTensor()
)

# allow for restricting classes to specific digits
from functools import reduce
def restrict_classes(dataset, classes):
    classes_membership_mask = reduce(lambda a,b: a|b,
                                     [dataset.targets == i for i in classes],
                                     torch.zeros(dataset.targets.size(), dtype=int))
    idx = torch.where(classes_membership_mask)
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]
    return dataset

# NOTE: you can change this list to select which digit classes are
# used from the datasets.
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print("Train data:")
print(restrict_classes(training_data, classes))
print()
print("Test data:")
print(restrict_classes(test_data, classes))

# NOTE: here, we configure the hyperparameters of the quantum layer.
# You must install the lightning.qubit device for cpu runs or
# lightning.gpu device for gpu runs. You can also set use_lightning=False
# to use the slow default.qubit backend.
nqubits = 4
nlayers = 2
use_lightning = False
use_gpu = False

# default to cpu pytorch ops
torch_device = "cpu"
if use_lightning:
    if use_gpu:
        qml_device = qml.device('lightning.gpu', wires=nqubits)
        # override to use gpu in pytorch
        torch_device = "cuda"
    else:
        qml_device = qml.device('lightning.qubit', wires=nqubits)
else:
    if use_gpu:
        raise RuntimeError("Cannot use gpu without also using lightning simulator.")
    qml_device = qml.device('qulacs.simulator', wires=nqubits)

# Here we define the quantum part of the model
@qml.qnode(qml_device, interface="torch")
def qnn_layer(inputs, weights):
    # encode inputs from previous layer
    # as rotations.
    for i in range(nqubits):
        qml.RX(inputs[i], wires=i)
    # place gates using the trainable weights
    # as parameters.
    for layer_index in range(nlayers):
        # place the trainable rotations
        for i in range(nqubits):
            qml.RY(weights[i + layer_index * nqubits], wires=i)
        # place the entangling gates
        for i in range(nqubits):
            j = (i + 1) % nqubits
            qml.CNOT(wires=(i, j))
    # now, return the pauli Z expectation values
    # on each qubit.
    return tuple(qml.expval(qml.PauliZ(i)) for i in range(nqubits))


class HybridClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # to convert image arrays to vectors
        self.flatten = nn.Flatten()
        # to reduce number of features for input to the qnn
        self.reduction_layer = nn.Linear(28*28, nqubits)
        # to map the qnn outputs to a class (some values are unused if
        # not using all 10 classes)
        self.output_layer = nn.Linear(nqubits, 10)

        # This is the randomly initialised weights tensor for the quantum layer.
        self.qnn_weights = torch.rand(nlayers * nqubits) * np.pi

    def forward(self, x):
        # transform image array to a vector
        x = self.flatten(x)
        # scale pixel values to [0, 1] range
        x = x / 255.0
        # apply classical dimensionality reduction layer
        x = self.reduction_layer(x)
        # apply pi*tanh activation to put data into the range from -pi
        # to pi
        x = torch.tanh(x) * torch.pi
        # apply the qnn layer to the input and weights.
        # older versions of pennylane do not support batches,
        # so we iterate through the batch and apply the qnn manually.
        batch_size = x.size(0)
        out = torch.empty((batch_size, nqubits), dtype=torch.float)
        for batch_index in range(batch_size):
            expval_tensors = qnn_layer(x[batch_index], self.qnn_weights)
            expval_floats = [t.item() for t in expval_tensors]
            out[batch_index] = torch.tensor(expval_floats)
        x = out
        # apply output layer to combine qnn outputs to 10 numbers
        x = self.output_layer(x)
        # just return outputs since we will use cross entropy loss in
        # training
        return x

model = HybridClassifier().to(torch_device)

# NOTE: the below code is all used to train the defined model
# and will be run when this file is loaded.
learning_rate = 1e-3
batch_size = 32
epochs = 30

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(torch_device)
        y = y.to(torch_device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    pred = None
    X = None
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(torch_device)
            y = y.to(torch_device)
            pred = model(X)
            predicted_label = torch.argmax(pred, dim=1)
            correct += torch.count_nonzero(predicted_label == y)
            test_loss += loss_fn(pred, y).item()

    print(f"Accuracy: {correct / (len(dataloader) * batch_size) * 100}")
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

start_time = time.time()

print(f"Before training\n--------------------------")
test_loop(test_dataloader, model, loss_fn)
for t in range(epochs):
    print(f"Epoch {t+1}\n--------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss = test_loop(test_dataloader, model, loss_fn)
    scheduler.step(test_loss)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training took {elapsed_time/60:.2f} minutes.")

print("Done!")
