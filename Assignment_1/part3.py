import torch
import torch.nn as nn
import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt


class MLPModel(nn.Module):
    """
        The class definition of the MLP model that inherits the nn.Module class.
    """

    def __init__(self, layerCount, hiddenNeurons, activationFunction):
        """
            Constructor of the class. Takes the total number of layers (excluding the input layer), total number of
            neurons in the hidden layers and the activation function as parameters.
        """
        super(MLPModel, self).__init__()
        self.layer1 = nn.Linear(784, hiddenNeurons)
        self.layer2 = nn.Linear(hiddenNeurons, 10)

        # Layer 3 is initially declared as None. If there are 2 hidden layers, this layer will be updated.
        self.layer3 = None

        if layerCount == 3:
            self.layer2 = nn.Linear(hiddenNeurons, hiddenNeurons)
            self.layer3 = nn.Linear(hiddenNeurons, 10)

        self.activation_function = activationFunction()

    def forward(self, x):
        """
            Feed-forward method of the MLP class. Takes the input dataset x as parameter.
        """
        hidden_layer_output = self.activation_function(self.layer1(x))

        # If there are 2 hidden layers, the output layer is once again updated.
        if self.layer3 is not None:
            hidden_layer_output = self.layer2(hidden_layer_output)
            output_layer = self.layer3(hidden_layer_output)

            return output_layer

        output_layer = self.layer2(hidden_layer_output)
        return output_layer


def accuracyCalculator(predictions, labels):
    """
        Helper function for calculating the accuracies. Compares the index values of the maximum values at each row
        of prediction tensor and the label tensor and returns the total number of matches.
    """
    max_probs = torch.argmax(predictions, dim=1)

    res = labels.eq(max_probs)

    return torch.count_nonzero(res).item()


# ============================== Data loading ===========================================
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
# ========================================================================================

# 12 configurations are used in the search and each configuration is declared as a dictionary. These dictionaries
# are stored in this array.
configs = [
    {'layer': 2, 'neuron': 16, 'func': nn.LeakyReLU},
    {'layer': 2, 'neuron': 16, 'func': nn.Sigmoid},
    {'layer': 2, 'neuron': 32, 'func': nn.LeakyReLU},
    {'layer': 2, 'neuron': 32, 'func': nn.Sigmoid},
    {'layer': 2, 'neuron': 64, 'func': nn.LeakyReLU},
    {'layer': 2, 'neuron': 64, 'func': nn.Sigmoid},
    {'layer': 3, 'neuron': 16, 'func': nn.LeakyReLU},
    {'layer': 3, 'neuron': 16, 'func': nn.Sigmoid},
    {'layer': 3, 'neuron': 32, 'func': nn.LeakyReLU},
    {'layer': 3, 'neuron': 32, 'func': nn.Sigmoid},
    {'layer': 3, 'neuron': 64, 'func': nn.LeakyReLU},
    {'layer': 3, 'neuron': 64, 'func': nn.Sigmoid},
]

loss_function = nn.CrossEntropyLoss()       # The cross-entropy loss is selected for the loss function

# Additional softmax function is defined from the torch.nn module to calculate probabilities of the predictions
soft_max_function = torch.nn.Softmax(dim=1)

i = 1

# The accuracy results of each run of each configuration will be stored as a matrix.
accuracies = []

# Traversing through every possible configuration
for config in configs:

    validation_accuracy = []

    # Running each configuration 10 times
    for j in range(1, 11):

        # For each run, a new model is defined
        model = MLPModel(layerCount=config['layer'], activationFunction=config['func'], hiddenNeurons=config['neuron'])

        # Adam optimizer is defined with the learning rate of 0.0001
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # Iteration (epoch) number is defined as 1000. Calculation of this value is explained in the report.
        ITERATION = 1000

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

                # Probability scores are calculated with the softmax function to calculate the accuracy
                probability_scores = soft_max_function(predictions)

                # Each iteration's accuracy scores are summed
                accuracy += accuracyCalculator(probability_scores, y_validation) / y_validation.size()[0]

                # Each loss is given as output for tracing
                print("Configuration: %d - Run : %d - Iteration: %d - Validation Loss: %f" % (
                    i, j, iteration, loss_value.item()))

        # The average of the accuracies are calculated and appended into the accuracy row corresponding row
        avgAccuracy = accuracy / ITERATION
        validation_accuracy.append(avgAccuracy)

    accuracies.append(validation_accuracy)

    i += 1


# Each configuration's confidence interval is calculated and given as output
i = 1
for acc in accuracies:
    n = np.array(acc)
    print(("The confidence interval of configuration %d: %f" + u"\u00B1" + ' %f') % (
        i, np.mean(n), np.std(n) / (1.96 * (len(acc)) ** 0.5)))
    i += 1

# The best configuration can be selected by the user by entering the index value as input
c = int(input('Pick a configuration for testing: '))
c -= 1

# The train and validation datasets along with their labels are combined
joined_x = torch.cat([x_train, x_validation])
joined_y = torch.cat([y_train, y_validation])

test_accuracies = []

for j in range(1, 11):

    # A new model is defined with the configuration that yiels the highest mean accuracy score
    model = MLPModel(layerCount=configs[c]['layer'], activationFunction=configs[c]['func'],
                     hiddenNeurons=configs[c]['neuron'])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    ITERATION = 1000

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

        # Test accuracies are calculated and stored
        test_accuracy = accuracyCalculator(test_predictions, y_test) / y_test.size()[0]

        test_accuracies.append(test_accuracy)

# The confidence interval for the testing accuracies is calculated and given as output
n = np.array(test_accuracies)
print(("The confidence interval of testing: %f " + u"\u00B1" + ' %f') % (
    np.mean(n), np.std(n) / (1.96 * (len(test_accuracies)) ** 0.5)))
