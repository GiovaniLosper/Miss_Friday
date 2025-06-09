import os
import math
import torch
from torch import nn

class Tokenizer:
    """Simple character-level tokenizer."""
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
    def encode(self, s):
        return [self.stoi[c] for c in s]
    def decode(self, tokens):
        return ''.join(self.itos[t] for t in tokens)


def load_data(path='data.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def get_batch(data, batch_size, block_size):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, self.heads, 3 * self.head_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = torch.softmax(att, dim=-1)
        out = att @ v
        out = out.reshape(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, ff_mult=4):
        super().__init__()
        self.attn = SelfAttention(embed_dim, heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_mult * embed_dim),
            nn.GELU(),
            nn.Linear(ff_mult * embed_dim, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, layers=2, heads=4, context=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, context, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, heads) for _ in range(layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.context = context

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.context, 'Sequence too long'
        x = self.embed(idx) + self.pos[:, :T]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx, max_new):
        for _ in range(max_new):
            idx_cond = idx[:, -self.context:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


def train(model, data, optimizer, steps, batch_size, block_size):
    model.train()
    for step in range(steps):
        xb, yb = get_batch(data, batch_size, block_size)
        logits = model(xb)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f'step {step} loss {loss.item():.4f}')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train or run a tiny transformer')
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--embed', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--context', type=int, default=128)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--mode', choices=['train', 'generate'], default='train')
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--tokens', type=int, default=50)
    args = parser.parse_args()

    text = load_data()
    tok = Tokenizer(text)
    encoded = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(encoded))
    train_data = encoded[:n]
    train_data = train_data
    vocab_size = len(tok.stoi)

    model = MiniGPT(vocab_size, args.embed, args.layers, args.heads, args.context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.mode == 'train':
        data = train_data
        train(model, data, optimizer, args.steps, args.batch, args.context)
        torch.save({'model': model.state_dict(), 'vocab': tok.stoi}, 'model.pt')
    else:
        ckpt = torch.load('model.pt')
        model.load_state_dict(ckpt['model'])
        tok.stoi = ckpt['vocab']
        tok.itos = {i: ch for ch, i in tok.stoi.items()}
        idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
        out = model.generate(idx, args.tokens)[0].tolist()
        print(tok.decode(out))


if __name__ == '__main__':
    main()
