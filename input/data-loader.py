import sys
import os
sys.path.append('./')
import numpy as np

from tokenizer import Tokenizer

tokenizer_name = input('Enter tokenizer filename: ')
dataset_name = input('Enter dataset name: ')

# check if path correct
if not os.path.exists(f'./tokenizer-dict/{tokenizer_name}'):
    print('Incorrect tokenizer')
    exit(0)
if not os.path.exists(f'./input/data/{dataset_name}/'):
    print('Incorrect dataset directory')
    exit(0)

# open training data
# TODO: fileloader
with open(f'./input/data/{dataset_name}/input.txt', encoding='utf-8', errors='ignore') as file:
    text = file.read()

# tokenizer initialization
tokenizer = Tokenizer()
with open(f'./tokenizer-dict/{tokenizer_name}', 'r') as file:
    dict = eval(file.read())
tokenizer.load_dict(dict)

data = tokenizer.encode(text)

# train and test splits
n = int(0.9*len(data)) # 90% test, 10% validation

train_data = np.array(data[:n], dtype=np.uint16)
val_data = np.array(data[n:], dtype=np.uint16)

train_data.tofile(f'./input/data/{dataset_name}/train.bin')
val_data.tofile(f'./input/data/{dataset_name}/val.bin')