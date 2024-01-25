import json
import re 
def read_json_file(file_name):
   
    with open("data/"+file_name+".json", 'r') as json_file:
        data = json.load(json_file)
    return data
def read_json_tokens(link):
   
    with open(link, 'r') as json_file:
        data = json.load(json_file)
    return data
def get_text(file_name,two_gramss):
    with open("texts/"+file_name + ".txt", 'r', encoding='utf-8') as file:
        content = file.read()
    content = content.split("\n")
    new_content = []
    for line in content:
        words_and_punctuation = re.findall(r'\b\w+\b', line)
        
        # Фильтрация пустых элементов
        non_empty_elements = [elem for elem in words_and_punctuation if elem.strip()]

        new_content.extend(non_empty_elements)
    result_content =[]
    for i in range(len(new_content)-1):
        b_gram = new_content[i] + " " + new_content[i+1]
        if b_gram in two_gramss:
            result_content.append(b_gram)
        else:
            result_content.append(new_content[i])
    return result_content