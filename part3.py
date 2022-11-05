import torch
import torch.nn as nn
import numpy as np
import pickle
import copy


class MLPModel(nn.Module):
    def __init__(self, layerCount, hiddenNeurons, activationFunction):
        super(MLPModel, self).__init__()
        self.layer1 = nn.Linear(784, hiddenNeurons)
        self.layer2 = nn.Linear(hiddenNeurons, 10)
        self.layer3 = None

        if layerCount == 3:
            self.layer2 = nn.Linear(hiddenNeurons, hiddenNeurons)
            self.layer3 = nn.Linear(hiddenNeurons, 10)

        self.activation_function = activationFunction()

    def forward(self, x):
        hidden_layer_output = self.activation_function(self.layer1(x))

        if self.layer3 is not None:
            hidden_layer_output = self.layer2(hidden_layer_output)
            output_layer = self.layer3(hidden_layer_output)

            return output_layer

        output_layer = self.layer2(hidden_layer_output)
        return output_layer


def accuracyCalculator(predictions, labels):
    max_probs = torch.argmax(predictions, dim=1)

    res = labels.eq(max_probs)

    return torch.count_nonzero(res).item()


x_train, y_train = pickle.load(open("data/mnist_train.data", "rb"))
x_validation, y_validation = pickle.load(open("data/mnist_validation.data", "rb"))
x_test, y_test = pickle.load(open("data/mnist_test.data", "rb"))

x_train = x_train / 255.0
x_train = x_train.astype(np.float32)

x_test = x_test / 255.0
x_test = x_test.astype(np.float32)

x_validation = x_validation / 255.0
x_validation = x_validation.astype(np.float32)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).to(torch.long)

x_validation = torch.from_numpy(x_validation)
y_validation = torch.from_numpy(y_validation).to(torch.long)

x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).to(torch.long)

models = [MLPModel(2, 32, nn.LeakyReLU),
          MLPModel(2, 32, nn.Sigmoid),
          MLPModel(2, 21, nn.LeakyReLU),
          MLPModel(2, 21, nn.Sigmoid)]

configs = [
    {'layer': 2, 'neuron': 32, 'func': nn.LeakyReLU, 'lr': 0.001},
    {'layer': 2, 'neuron': 32, 'func': nn.LeakyReLU, 'lr': 0.0001},
    {'layer': 2, 'neuron': 32, 'func': nn.Sigmoid, 'lr': 0.001},
    {'layer': 2, 'neuron': 32, 'func': nn.Sigmoid, 'lr': 0.0001},
    {'layer': 2, 'neuron': 21, 'func': nn.LeakyReLU, 'lr': 0.001},
    {'layer': 2, 'neuron': 21, 'func': nn.LeakyReLU, 'lr': 0.0001},
    {'layer': 2, 'neuron': 21, 'func': nn.Sigmoid, 'lr': 0.001},
    {'layer': 2, 'neuron': 21, 'func': nn.Sigmoid, 'lr': 0.0001},
    {'layer': 3, 'neuron': 32, 'func': nn.LeakyReLU, 'lr': 0.001},
    {'layer': 3, 'neuron': 32, 'func': nn.LeakyReLU, 'lr': 0.0001},
    {'layer': 3, 'neuron': 32, 'func': nn.Sigmoid, 'lr': 0.001},
    {'layer': 3, 'neuron': 32, 'func': nn.Sigmoid, 'lr': 0.0001},
    {'layer': 3, 'neuron': 21, 'func': nn.LeakyReLU, 'lr': 0.001},
    {'layer': 3, 'neuron': 21, 'func': nn.LeakyReLU, 'lr': 0.0001},
    {'layer': 3, 'neuron': 21, 'func': nn.Sigmoid, 'lr': 0.001},
    {'layer': 3, 'neuron': 21, 'func': nn.Sigmoid, 'lr': 0.0001}
]

loss_function = nn.CrossEntropyLoss()

soft_max_function = torch.nn.Softmax(dim=1)

i = 1

accuracies = []

for config in configs:

    validation_accuracy = []
    max_accuracy = -1

    for j in range(1, 11):

        model = MLPModel(layerCount=config['layer'], activationFunction=config['func'], hiddenNeurons=config['neuron'])

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

        ITERATION = 150

        accuracy = 0

        for iteration in range(1, ITERATION + 1):
            optimizer.zero_grad()
            predictions = model(x_train)

            loss_value = loss_function(predictions, y_train)

            loss_value.backward()
            optimizer.step()

            with torch.no_grad():
                predictions = model(x_validation)
                loss_value = loss_function(predictions, y_validation)

                probability_scores = soft_max_function(predictions)

                accuracy += accuracyCalculator(probability_scores, y_validation) / y_validation.size()[0]

                print("Configuration: %d - Run : %d - Iteration: %d - Validation Loss: %f" % (
                    i, j, iteration, loss_value.item()))

        avgAccuracy = accuracy / ITERATION

        validation_accuracy.append(avgAccuracy)

    accuracies.append(validation_accuracy)

    i += 1

i = 1
for acc in accuracies:
    n = np.array(acc)
    print(("The confidence interval of configuration %d: %f" + u"\u00B1" + ' %f') % (
        i, np.mean(n), np.std(n) / (1.96 * (len(acc)) ** 0.5)))
    i += 1

c = int(input('Pick a configuration for testing: '))
c -= 1

joined_x = torch.cat([x_train, x_validation])
joined_y = torch.cat([y_train, y_validation])

test_accuracies = []

for j in range(1, 11):

    model = MLPModel(layerCount=configs[c]['layer'], activationFunction=configs[c]['func'],
                     hiddenNeurons=configs[0]['neuron'])

    optimizer = torch.optim.Adam(model.parameters(), lr=configs[c]['lr'])

    ITERATION = 150

    for iteration in range(1, ITERATION + 1):
        optimizer.zero_grad()
        predictions = model(joined_x)

        loss_value = loss_function(predictions, joined_y)

        loss_value.backward()
        optimizer.step()

        with torch.no_grad():
            predictions = model(joined_x)
            loss_value = loss_function(predictions, joined_y)

            print("Test Run : %d - Iteration: %d - Loss: %f" % (
                j, iteration, loss_value.item()))

    with torch.no_grad():
        test_predictions = model(x_test)

        test_accuracy = accuracyCalculator(test_predictions, y_test) / y_test.size()[0]

        test_accuracies.append(test_accuracy)

n = np.array(test_accuracies)
print(("The confidence interval of testing: %f " + u"\u00B1" + ' %f') % (
    np.mean(n), np.std(n) / (1.96 * (len(test_accuracies)) ** 0.5)))
