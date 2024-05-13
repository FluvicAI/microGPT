import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        #compute attention scores
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # weighted aggregation of values
        v = self.value(x)
        out = wei @ v
        return out

class FeedForward(nn.Module):
    """MLP layer"""
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4), 
            nn.ReLU()
        ) # inner feedforward should have fan_out of 4*n_embed
    
    def forward(self, x):
        return self.net(x)

class MultiheadedAttention(nn.Module):
    def __init__(self, n_head, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed, block_size, dropout) for _ in range(n_head)])

    def forward(self,x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
    
class Block(nn.Module):
    def __init__(self, n_head, n_embed, block_size, dropout):
        super().__init__()
        head_size = n_embed//n_head   # n_head = amount of heads, head_size = dimension of key, query and val
        # multiheaded self attention block
        self.sa = MultiheadedAttention(n_head, head_size, n_embed, block_size, dropout)
        # projection matrix for adding to residual connection
        self.proj_sa = nn.Linear(n_embed, n_embed)
        self.proj_ffwd = nn.Linear(4 * n_embed, n_embed) # inner feedforward has dimensionality of 4*n_embed
        # MLP with ReLU non-linearity
        self.ffwd = FeedForward(n_embed)
        # normalisation over batch AND time dimension -> ensures stability for deep networks
        self.ln1 = nn.LayerNorm(n_embed) #pre SA layernorm
        self.ln2 = nn.LayerNorm(n_embed) #pre ffwd layernorm

        # dropout -> decreases overfitting
        self.dropout_sa = nn.Dropout(dropout)
        self.dropout_ffwd = nn.Dropout(dropout)

    def forward(self, x):
        # add residual connections
        x = self.dropout_sa(self.proj_sa(self.sa(self.ln1(x)))) + x
        x = self.dropout_ffwd(self.proj_ffwd(self.ffwd(self.ln2(x)))) + x

        return x

class GPTmodel(nn.Module):
    def __init__(self, n_embed, vocab_size,  block_size, n_head, n_transformer_blocks, dropout, device):
        super().__init__()
        # loading hyperparameters
        self.n_embed = n_embed
        self.vocab_size = vocab_size
        self.block_size =  block_size
        self.n_head = n_head
        self.dropout = dropout
        self.device = device
        self.n_transformer_blocks = n_transformer_blocks

        #embed tokens (input = one_hot vector)
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embed)

        # superpose a location embedding
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embed) # 4 self attention blocks, with dimension 8 -> concatenated back to 32 (n_embed) = 4 attention channels

        # transformer blocks
        self.transformer_blocks = nn.Sequential(*[Block(self.n_head, self.n_embed, self.block_size, self.dropout) for _ in range(self.n_transformer_blocks)])
        self.ln_f = nn.LayerNorm(self.n_embed) # transformer blocks + last layernorm

        # final Linear layer that brings back down to vocab dimension (logit dimension) (used in mutinomial for sampling)
        self.lm_head = nn.Linear(self.n_embed, self.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) #(B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) #(T,C)
        x = tok_emb + pos_emb # location embedding superinposed
        x = self.transformer_blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)

            # targets of shape (B,T) -> (B*T)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets) # expects logits of structure (B, C, T)

        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            # only take context as last block_size characters
            idx_cond = idx[:, -self.block_size:]
            # get predictions
            logits, loss = self(idx_cond)
            # get last token
            #TODO: fix temp 0 bug
            logits = logits[: , -1, :] / temperature #(B,C)
            # softmax
            probs = F.softmax(logits, dim=1)
        
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)

            idx = torch.cat((idx,idx_next), dim=1) # (B, T+1)
        
        return idx