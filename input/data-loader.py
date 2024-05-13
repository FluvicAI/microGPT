from tokenizer import Tokenizer

tokenizer_name = input('Enter tokenizer filename: ')
dataset_name = input('Enter dataset_name: ')

# open training data
# TODO: fileloader
with open('./input/input.txt', encoding='utf-8', errors='ignore') as file:
    text = file.read()

# tokenizer initialization
tokenizer = Tokenizer()
with open(f'./tokenizer-dict/{tokenizer_name}') as file:
    dict = eval(file.read())
tokenizer.load_dict(dict)

data = None

# train and test splits
n = int(0.9*len(data)) # 90% test, 10% validation

train_data = data[:n]
val_data = data[n:]