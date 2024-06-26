import torch
import time
import datetime
import numpy as np
import os

from model import GPTmodel

# hyperparameters ------
batch_size = 64
block_size = 256 # increase??
max_iter = 2000 
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384   # should be multiple of n_heads ! (lower n_embed because of overfitting?)
n_transformer_blocks = 8
n_heads = 6
dropout = 0.2 # dropout higher?, trainingset probably to small
# ----------------------

print('---------------------------------------------')
print('Using device: ' + str(device))
print('----------------------------------------------')

train_further = input('Train further upon existing model? ')
if train_further == 'y':
    train_further = True
    old_model = input('Old model filename: ')

model_name = input('New model filename [ENTER to skip]: ') # default: timestamp
data_dir = input('Training set path: ') # default './data'

if data_dir is None:
    data_dir = './input/data'
else: 
    data_dir = f'./input/data/{data_dir}'

ts = time.time()

# SEED
torch.manual_seed(1337)

# initialize tokenizer (just for vocab size)
tokenizer_name = input('Tokenizer filename: ')
with open(f'./tokenizer-dict/{tokenizer_name}') as file:
    dict = eval(file.read())

vocab_size = dict['vocab_size']
del dict # no use for dict anymore

if train_further:
    # training further upon other model, overwrite hyperparameters
    weightsfile = f"./weights/nanoGPT_weights_{old_model}.pth"
    paramsfile = f"./model_params/nanoGPT_params_{old_model}.txt"

    with open(paramsfile, 'r') as file:
        modelParams = eval(file.read())

    modelState = torch.load(weightsfile, map_location=device)

    block_size = modelParams['block_size']
    n_embed = modelParams['n_embed']
    n_heads = modelParams['n_heads']
    n_transformer_blocks = modelParams['n_transformer_blocks']
    vocab_size = modelParams['vocab_size']
    tokenizer_name = modelParams['tokenizer_name']

model = GPTmodel(n_embed, vocab_size, block_size, n_heads, n_transformer_blocks, dropout, device)
model.train()

m = model.to(device)

if train_further:
    model.load_state_dict(modelState)

print('Starting Training....')
print(f"Parameters: {sum(p.nelement() for p in model.parameters())}")
    
@torch.no_grad()
def estimate_loss():
    out = {}
    #setting model to eval phase
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        #resetting model to train phase
    model.train()
    return out

def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+block_size+1]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

def write_checkpoint():
    torch.save(model.state_dict(), f'./weights/nanoGPT_weights_{model_name}_checkpoint.pth')
    modelParams = {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_embed': n_embed,
        'n_transformer_blocks': n_transformer_blocks,
        'n_heads': n_heads,
        'vocab_size': vocab_size,
        'tokenizer_name': tokenizer_name
    }
    with open(f'./model_params/nanoGPT_params_{model_name}_checkpoint.txt','w') as file:
        file.write(str(modelParams))

    print('Succesfully saved checkpoint')
    return

##Training
# create optimiser (AdamW)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(max_iter):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        try:
            write_checkpoint()
        except:
            print('Error while saving checkpoint')

        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, Elapsed time: {(time.time() - ts):.1f} s")

    xb, yb = get_batch('train')
    
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# save parameters
current_time = datetime.datetime.now()
timestamp = f'{current_time.day}-{current_time.month}-{current_time.year}_{current_time.hour}-{current_time.minute}-{current_time.second}'
if model_name is None:
    model_name = timestamp

modelParams = {
    'vocab_size': vocab_size,
    'block_size': block_size,
    'n_embed': n_embed,
    'n_transformer_blocks': n_transformer_blocks,
    'n_heads': n_heads,
    'vocab_size': vocab_size,
    'tokenizer_name': tokenizer_name
}

torch.save(model.state_dict(), f'./weights/nanoGPT_weights_{model_name}.pth')
with open(f'./model_params/nanoGPT_params_{model_name}.txt','w') as file:
    file.write(str(modelParams))

# TODO: delete checkpoint

print('\n')
print('TRAINING SUMMARY ----------------------------------------------')
print(f"Final loss: train: {losses['train']:.4f} | val: {losses['val']:.4f}")
print(f"Training Time: {(time.time() - ts):.1f} s")
print(f"Parameters: {sum(p.nelement() for p in model.parameters())}")
print(f"Weights exported to: ./weights/nanoGPT_weights_{model_name}.pth")
print('---------------------------------------------------------------')


#TO DO FOR NEXT VERSION:
# - tokenizer: add manual tokens + special tokens functions
# - graphical logs
# - functionality to stop training and save parameters
# - better error handling