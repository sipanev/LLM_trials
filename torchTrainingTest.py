import torch
import torch.nn as nn
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

print('Hello, hello!')

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 10
# ------------

torch.manual_seed(5007)

# Load the tokens
tok = TokenizerX('tokens.txt')
tokens = tok.GetTokenList()
print(len(tokens),'tokens loaded')

with open('K:/ProjectData/texts/TheGiftOfTheMagi.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print('text len:', len(text))

vocab_size = len(tokens)

# create a mapping from characters to integers
#stoi = { ch:i for i,ch in enumerate(chars) }
#itos = { i:ch for i,ch in enumerate(chars) }
#encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
#decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# encoder: take a string, output a list of integers
encode = lambda s: tok.TextToNumbers(s, True)
# decoder: take a list of integers, output a string
decode = lambda l: ' '.join([tok.NumberToToken(i) for i in l])

print('Training...')
# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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
model.Save('model.'+str(max_iters)+'.bin')

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))