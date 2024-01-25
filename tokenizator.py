import random
import json
import numpy as np
import re 

def tokenizer(file_name):
    with open("texts/"+file_name + ".txt", 'r', encoding='utf-8') as file:
        content = file.read()
    
    tokens={}

    two_gramss = {}

    content = content.split("\n")
    new_content = []
    for line in content:
        words_and_punctuation = re.findall(r'\b\w+\b', line)
        
        # Фильтрация пустых элементов
        non_empty_elements = [elem for elem in words_and_punctuation if elem.strip()]

        new_content.extend(non_empty_elements)
    # print(new_content)
    for i in range(len(new_content)//2):
        string = new_content[i]+" "+new_content[i+1]
        if string in two_gramss:
            two_gramss[new_content[i]+" "+new_content[i+1]] += 1
        else:
            two_gramss[new_content[i]+" "+new_content[i+1]] = 1
    new_two_grams = {}
    for key, value in two_gramss.items():
        if value >= 5:
            new_two_grams[key] = value
    two_gramss =new_two_grams
    for i in range(len(new_content)-1):
        t_g =new_content[i] + " " + new_content[i+1]
        is_t_g = True
        for j in two_gramss:
            if t_g == j:
                is_t_g = False
                if t_g in tokens:
                    tokens[t_g]+=1
                else:
                    tokens[t_g]=1
        if is_t_g:
            if new_content[i] in tokens:
                tokens[new_content[i]]+=1
            else:
                tokens[new_content[i]]=1
    if new_content[-1] in tokens:
        tokens[new_content[-1]]+=1
    else:
        tokens[new_content[-1]]=1

    new_tokens ={}
    count=0
    for key, value in tokens.items():
        if value >=5:
            new_tokens[key] = [value,count]
            count+=1
    tokens = new_tokens
    print(len(two_gramss))
    print(len(tokens))
    # print(tokens)
    file_name = 'data/data_count.txt'

    with open(file_name, 'r') as file:
        number = int(file.read())

    number += 1

    with open(file_name, 'w') as file:
        file.write(str(number))


    file_name = 'data/data_'+str(number)+'.json'
    with open(file_name, 'w') as json_file:
        json.dump(tokens, json_file)
    file_name = 'data/data_'+str(number)+'_two_gramms.json'
    with open(file_name, 'w') as json_file:
        json.dump(two_gramss, json_file)
                




tokenizer("Hp_half_6")  
