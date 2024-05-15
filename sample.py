import torch
import time
from tokenizer import Tokenizer

from model import GPTmodel

# hyperparameters ------
device =  'cuda' if torch.cuda.is_available() else 'cpu'
# ----------------------

model_name = input('Enter model name: ')

# load parameters
weightsfile = f"./weights/nanoGPT_weights_{model_name}.pth"
paramsfile = f"./model_params/nanoGPT_params_{model_name}.txt"

with open(paramsfile, 'r') as file:
    modelParams = eval(file.read())

modelState = torch.load(weightsfile, map_location=device)

block_size = modelParams['block_size']
n_embed = modelParams['n_embed']
n_heads = modelParams['n_heads']
n_transformer_blocks = modelParams['n_transformer_blocks']
vocab_size = modelParams['vocab_size']
tokenizer_name = modelParams['tokenizer_name']

# tokenizer initialization
tokenizer = Tokenizer()
with open(f'./tokenizer-dict/{tokenizer_name}.txt', 'r') as file:
    dict = eval(file.read())
tokenizer.load_dict(dict)

model = GPTmodel(n_embed, vocab_size, block_size, n_heads, n_transformer_blocks, 0, device)
m = model.to(device)

model.load_state_dict(modelState)

model.eval()

def sample(start_context, length, temp):
    context_enc = tokenizer.encode(start_context)
    context = torch.zeros((1,len(context_enc)), dtype=torch.long, device=device)
    context[0,:] = torch.tensor(context_enc)
    return tokenizer.decode(model.generate(context, max_new_tokens=length, temperature=temp)[0].tolist())

while True:
    prompt = bytes(input('ENTER PROMPT: '), "utf-8").decode("unicode_escape")
    length = input('ENTER MAX GENERATION LENGTH: ')
    set_seed = input('SET SEED? y/N ').strip()
    temp = input('SET TEMPERATURE [1.0]: ')
    if set_seed == 'y':
        seed = input('ENTER SEED: ')
        if seed == '':
            seed = 1337
        torch.manual_seed(int(seed))

    if temp == '':
        temp = 1.0

    ts = time.time()


    print('------------------------------------------------')
    print(sample(prompt, int(length), float(temp)))
    print('------------------------------------------------')
    print(f"Inference time: {(int(length)/(time.time() - ts)):.1f} token/s")
    repeat = input('Press ENTER to exit, or type r to generate another prompt: ')
    if repeat != 'r':
        break