import json
import math
import random
import numpy as np


class NeuronLayer:
    def __init__(self, input_count, neuron_count, activation_function, learning_rate):
        self.neuron_count = neuron_count
        self.input_count = input_count
        self.learning_rate = learning_rate

        self.activation_function = activation_function
        self.W = [[random.uniform(-0.5, 0.5) for _ in range(input_count)] for _ in range(neuron_count)]
        self.bias = [0.01 * (random.random() - 0.5) for _ in range(neuron_count)]

    def forward(self, inputs):
        result = [0] * self.neuron_count
        for i in range(self.neuron_count):
            result[i] = sum(x * w + b for x, w, b in zip(inputs, self.W[i], [self.bias[i]] * len(inputs)))
            result[i] = self.activation_function(result[i])
        return result

    def backward(self, e, inputs):
        result = [[] for _ in range(self.neuron_count)]
        for i in range(self.neuron_count):
            self.bias[i] -= self.learning_rate * e[i]
            for j in range(self.input_count):
                result[i].append(e[i] * self.W[i][j])
                self.W[i][j] += self.learning_rate * result[i][j] * inputs[j]

        err = [sum(result[j][i] for j in range(self.neuron_count)) for i in range(self.input_count)]
        return err

    def save_parameters_to_json(self, file_path):
        try:
            json_data = json.dumps(self.W, indent=2)
            with open(file_path, 'w') as file:
                file.write(json_data)
        except Exception as ex:
            print(f"Ошибка при сохранении параметров слоя: {ex}")

    def load_parameters_from_json(self, file_path):
        try:
            with open(file_path, 'r') as file:
                json_data = file.read()
                self.W = json.loads(json_data)
        except Exception as ex:
            print(f"Ошибка при загрузке параметров слоя: {ex}")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

