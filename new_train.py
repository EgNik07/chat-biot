from defs.func import *
from defs.NeuronLayer import *
import random
import numpy as np

def unique_random():
    used_numbers = set()
    while True:
        num = random.random()
        if num not in used_numbers:
            used_numbers.add(num)
            return num

tokens = read_json_file("data_6")
two_gramss = read_json_file("data_6_two_gramms")
text = get_text("HP_half_3", two_gramss)

tokens_len = len(tokens)
tokens_indexs = {key: unique_random() for key in tokens}

learning_rate = 0.001

l1 = NeuronLayer(10, 128, sigmoid, learning_rate)
l2 = NeuronLayer(128, tokens_len, sigmoid, learning_rate)

l11 = NeuronLayer(tokens_len, 128, sigmoid, learning_rate)
l12 = NeuronLayer(128, tokens_len, sigmoid, learning_rate)

data_numers = [tokens[j][1] for j in text if j in tokens]

new_text = []
for j in text:
    if j in tokens:
        new_text.append(j)
text = new_text

inputs = [tokens_indexs[text[i]] for i in range(9)]
epochs = 1
text_size = len(text)

for _ in range(epochs):
    for j in range(9, text_size - 1):
        inputs.append(tokens_indexs[text[j]])

        x1 = l1.forward(inputs)
        x2 = l2.forward(x1)

        embeding_vector = np.zeros(tokens_len)
        embeding_vector[data_numers[j]] = x2[data_numers[j]]

        x11 = l11.forward(embeding_vector)
        x12 = l12.forward(x1)

        result = [x2[k] - (data_numers[j+1] == k) for k in range(tokens_len)]

        e1 = l2.backward(result, x1)
        e2 = l1.backward(e1, inputs)

        inputs = inputs[1:]

print("Training completed.")

for _ in range(10):
    r = random.randint(9, text_size)
    test_text = [tokens_indexs[text[i]] for i in range(r-9, r)]

    x1 = l1.forward(test_text)
    x2 = l2.forward(x1)

    big_index = np.argmax(x2)
    
    for key, value in tokens.items():
        if value[1] == big_index:
            print("Prediction: " + key)
            break
