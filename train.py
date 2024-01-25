from defs.func import * 
from defs.NeuronLayer import * 
import os
tokens = read_json_file("data_10")
two_gramss = read_json_file("data_10_two_gramms")
text = get_text("HP_half_6",two_gramss)



tokens_len=len(tokens)
tokens_indexs = {}
for key, value in tokens.items():
    while True:
        num = random.random()
        is_new_num=True
        for i_key, i_value in tokens_indexs.items():
            if i_value == num:
                is_new_num = False
                break
        if is_new_num:
            tokens_indexs[key] = num
            break
learning_rate = 0.0001
# embedding

l1 = NeuronLayer(10, 128, sigmoid,learning_rate)
l2 = NeuronLayer(128, tokens_len, sigmoid,learning_rate)

# embedding

# layers nn 

l11 = NeuronLayer(tokens_len, 128, sigmoid,learning_rate)
l12 = NeuronLayer(128, tokens_len, sigmoid,learning_rate)

# layers nn 
data_numers = []
new_text = []
for j in text:
    if j in tokens:
        new_text.append(j)
text = new_text
for j in text:
    data_numers.append(tokens[j][1])

inputs=[]
for i in range(9):
    inputs.append(tokens_indexs[text[i]])

epochs =10
text_size = len(text)
for i in range(epochs):
    for j in range(9,text_size-1):

        inputs.append(tokens_indexs[text[j]])

        x1 = l1.forward(inputs)
        x2 = l2.forward(x1)

        embeding_vector = np.zeros(tokens_len)
        for h in inputs:
            embeding_vector[data_numers[j]] = x2[data_numers[j]]

        x11 = l11.forward(embeding_vector)
        x12 = l12.forward(x1)

        result= []
        true_res = np.zeros(tokens_len)
        true_res[data_numers[j+1]]

        for k in range(len(true_res)):
            result.append(x2[k] - true_res[k])
        # train

        e1 = l2.backward(result,x1)
        e2 = l1.backward(e1,inputs)

        inputs=inputs[1:]

        print(" epochs: ",epochs," / ", i," text:",text_size," / ",j, " word:", text[j] )
for e in range(10):
    r = random.randint(9, text_size)
    test_text = []
    for i in range(r-9,r):
        test_text.append(tokens_indexs[text[i]])


    x1 = l1.forward(test_text)
    x2 = l2.forward(x1)
    big_num =0
    big_index =0
    for i in range(len(x2)):
        if x2[i]> big_num:
            big_num=x2[i]
            big_index=i
 
    for key, value in tokens.items():
        if value[1] == big_index:
            print("res:" + key)
            break

file_name = 'models/models_count.txt'

with open(file_name, 'r') as file:
    number = int(file.read())

number += 1

with open(file_name, 'w') as file:
    file.write(str(number))
m_dir = "./models/model_"+str(number)

# Check if the folder doesn't exist and create it
if not os.path.exists(m_dir):
    os.makedirs(m_dir)
l1.save_parameters_to_json(m_dir + "/l1.json")
l2.save_parameters_to_json(m_dir + "/l2.json")

l11.save_parameters_to_json(m_dir + "/l11.json")
l12.save_parameters_to_json(m_dir + "/l12.json")

file_name = m_dir+'/tokens.json'
with open(file_name, 'w') as json_file:
    json.dump(tokens, json_file)
     
    