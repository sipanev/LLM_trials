import sys
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from TokenizerX import *
from BigramLanguageModel import *

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def ChooseTrainAndValidationData0(dataArr):
    n = int(0.9 * len(dataArr))  # first 90% will be train, rest val
    trainData = dataArr[:n]
    valData = dataArr[n:]
    return trainData, valData

def ChooseTrainAndValidationData(dataArr):
    train, test = train_test_split(dataArr, test_size=0.1)
    return train, test


# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = int(sys.argv[2])
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 10
# ------------

filename = sys.argv[1]

print('Loading model from "',filename,'"...')
model = BigramLanguageModel.Load(filename)
m = model.to(device)

# Load the tokens
print('Loading the tokens/tokenizer ...')
tok = TokenizerX('tokens.txt')
tokens = tok.GetTokenList()
print(len(tokens),'tokens loaded')

with open('K:/ProjectData/texts/TheGiftOfTheMagi.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print('text len:', len(text))

# encoder: take a string, output a list of integers
encode = lambda s: tok.TextToNumbers(s, True)
# decoder: take a list of integers, output a string
decode = lambda l: ' '.join([tok.NumberToToken(i) for i in l])

# Find a random value?
torch.manual_seed(123)

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
train_data, val_data = ChooseTrainAndValidationData(data)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print('Training continued ...')
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    # sample a batch of data
    xb, yb = get_batch('train')
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model
model.Save(filename+'+'+str(max_iters))

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))