from tokenizer import Tokenizer

tokenizer_input = input('Enter tokenizer input text name: ')
vocab_size = input('Enter tokenizer vocab size: ')
tokenizer_name = input('Enter tokenizer name: ')

tokenizer = Tokenizer()

# TODO 
# here implementation for special tokens/manual tokens (can also be done after running this script though)

tokenizer.process_input(f'./input/tokenizer/{tokenizer_input}.txt', vocab_size, f'./tokenizer-dict/{tokenizer_name}.txt')