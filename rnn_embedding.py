import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import shuffle
import numpy as np

BATCH_SIZE = 3

class RNN(nn.Module):
    def __init__(self, vocab, hidden, max_len):
        super(RNN, self).__init__()
        self.vocab_size = len(vocab.keys())
        self.hidden = hidden
        self.vocab = vocab
        self.rnn = nn.RNN(input_size=self.vocab_size+1, hidden_size=hidden, bidirectional=True)
        self.decoder = nn.Linear(in_features=hidden*2, out_features=self.vocab_size+1)
        self.drop = nn.Dropout(0.2)
        self.embeddings = nn.Embedding.from_pretrained(torch.eye(self.vocab_size + 1))
        self.max_len = max_len

    def embed(self, batch, out=False):
        # word = word.split(' ')
        # emb = torch.zeros((len(word), 1, self.vocab_size))
        # for i, w in enumerate(word):
        #     emb[i][0][self.vocab[w]] = 1.0
        # print(([[self.vocab[w] for w in word.split(' ')] for word in batch]))
        # emb = self.embeddings(torch.tensor([[self.vocab[w] for w in word.split(' ')] for word in batch]))
        # return emb
        if out:
            emb = torch.zeros((len(batch), max([len(b) for b in batch]) + 3)).long()
        else:
            emb = torch.zeros((len(batch), max([len(b) for b in batch]) + 3, self.vocab_size+1))
        for i, word in enumerate(batch):
            word = word.split(' ')
            if out:
                emb[i][:len(word) + 3] = torch.tensor([self.vocab_size] + [self.vocab[w] for w in word] + [self.vocab_size, self.vocab_size])

            else:
                emb[i][:len(word) + 3] =  self.embeddings(torch.tensor([self.vocab_size, self.vocab_size] + [self.vocab[w] for w in word] + [self.vocab_size]))

        if out:
            emb = emb
        else:
            emb = emb.permute(1, 0, 2)

        return emb

    def forward(self, x):
        # x, hidden = self.rnn(x, (torch.randn(2, x.shape[1], self.hidden), torch.randn(2, x.shape[1], self.hidden)))
        x, hidden = self.rnn(x, torch.randn(2, x.shape[1], self.hidden))
        x = self.drop(x)
        x = self.decoder(x)
        return F.log_softmax(x, dim=2).permute(1,2,0)

    def get_vecs(self):
        vecs = np.zeros((self.vocab_size, self.hidden*2))
        for i, v in enumerate(self.vocab):
            emb = self.embed([v])
        #     # _, (h, _) = self.rnn(emb[2][0].unsqueeze(0).unsqueeze(0), (torch.zeros(2,1, self.hidden), torch.zeros(2, 1, self.hidden)))
            _, h = self.rnn(emb[2][0].unsqueeze(0).unsqueeze(0), torch.randn(2,1, self.hidden))
            vecs[i] = torch.cat((h[1][0], h[0][0])).detach().numpy()
        return vecs

        # use outputs as vecs
        # vecs = np.zeros((self.vocab_size, self.hidden*2))
        # for i, v in enumerate(self.vocab):
        #     emb = self.embed([v])
        #     x, _ = self.rnn(emb[2][0].unsqueeze(0).unsqueeze(0), torch.randn(2,1, self.hidden))
        #     vecs[i] = x[0][0].detach().numpy()
        # return vecs

def train_epoch(model,optimizer, criterion,data):
    model.train()
    shuffle(data)
    for i, w in enumerate(batch(data, BATCH_SIZE)):
        optimizer.zero_grad()
        emb = model.embed(w)
        emb_out = model.embed(w, out=True)
        out = model.forward(emb)
        loss = criterion(out, emb_out)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('{:f}%, loss: {:f}'.format((i/(len(data)/BATCH_SIZE))*100, loss), end='\r')

def evaluate(model, criterion, data):
    with torch.no_grad():
        model.eval()
        total_loss = 0.
        batches = 0
        # for w in data:
        #     emb = model.embed([w])
        #     emb_out = model.embed([w], out=True)
        #     out = model.forward(emb)
        #     total_loss += criterion(out, emb_out).item()
        # return total_loss / len(data)
        for i, w in enumerate(batch(data, BATCH_SIZE)):
            if i % 100 == 0:
                print('{:f}%'.format((i/(len(data)/BATCH_SIZE))*100), end='\r')
            emb = model.embed(w)
            emb_out = model.embed(w, out=True)
            out = model.forward(emb)
            total_loss += criterion(out, emb_out).item()
            batches += 1
        return total_loss / batches

def batch(data, batch_size):
    l = len(data)
    for i in range(0, l, batch_size):
        yield data[i:min(i + batch_size, l)]

def get_rnn_embeddings(data, vocab):
    HIDDEN_SIZE = len(vocab.keys())
    # HIDDEN_SIZE = 5

    max_len = max([len(d) for d in data])
    model = RNN(vocab, HIDDEN_SIZE, max_len)
    criterion = nn.NLLLoss()
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=0.001,
        # weight_decay=1e-6,
    )
    prev_eval = np.inf
    eval = 10
    epochs = 0
    best_eval = np.inf
    best_vecs = None

    # while prev_eval > eval and epochs < max_epochs:
    while epochs < 1000:
        prev_eval = eval
        train_epoch(model, optimizer, criterion, data)
        eval = evaluate(model, criterion, data)
        epochs += 1
        print('Epoch {} eval loss: {}'.format(epochs, eval))
        vecs = model.get_vecs()
        # vecs_hidden = model.get_vecs_hidden()
        if eval < best_eval:
            best_eval = eval
            best_vecs = vecs
        np.save('{}_{}.npy'.format('rnn_vecs/rnn', epochs), vecs)
        # np.save('{}_hidden_{}.npy'.format(vecspath, epochs), vecs_hidden)

    return best_vecs
