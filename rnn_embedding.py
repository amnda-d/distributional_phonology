import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import shuffle
import numpy as np

torch.manual_seed(0)
np.random.seed(0)
torch.set_deterministic(True)

class ContextRNN(nn.Module):
    def __init__(self, vocab, hidden, window_size=1):
        super(ContextRNN, self).__init__()
        self.vocab_size = len(vocab.keys())
        self.vocab = vocab
        self.window_size = window_size
        self.hidden = hidden
        self.embeddings = nn.Embedding.from_pretrained(torch.eye(self.vocab_size + 1))

        self.rnn = nn.RNN(
            input_size=self.vocab_size+1,
            hidden_size=hidden,
            bidirectional=True,
        )

        self.lin = nn.Linear(
            in_features=hidden*2,
            out_features=self.vocab_size
        )

        self.decoder = nn.Linear(
            in_features=self.vocab_size,
            out_features=(self.vocab_size+1)*(window_size*2)
        )

        self.drop = nn.Dropout(0.25)

    def forward(self, word):
        # input shape:  (seq_len, batch, input_size)
        x = self.word_to_in(word).unsqueeze(1)
        x, hidden = self.rnn(x, self.init_hidden(x))
        x = torch.tanh(x)
        x = self.drop(x)
        x = self.lin(x)
        x = torch.tanh(x)
        x = self.decoder(x)
        # shape: (seq_len, batch, vocab_size, window_size*2)
        x = x.view((x.shape[0], x.shape[1], self.vocab_size+1, self.window_size*2))
        # shape: (batch, vocab, len, window)
        return F.log_softmax(x, dim=2).permute(1, 2, 0, 3).contiguous()

    def init_hidden(self, x):
        return torch.randn((2, x.shape[1], self.hidden))

    def word_to_in(self, word):
        word = [w for w in word.strip().split(' ') if w != '']
        return self.embeddings(torch.tensor([self.vocab_size if w == '#' else self.vocab[w] for w in word]))

    def word_to_out(self, word):
        word = [w for w in word.strip().split(' ') if w != '']
        padded = ['#']*self.window_size + word + ['#']*self.window_size

        out = torch.tensor([[self.vocab_size if padded[x] == '#' else self.vocab[padded[x]] for x in list(range(i-self.window_size, i)) + list(range(i+1, self.window_size+i+1))] for i in range(self.window_size, len(word)+self.window_size)]).long()

        return out.unsqueeze(0)

    def get_vecs(self):
        with torch.no_grad():
            self.eval()
            # vecs = np.zeros((self.vocab_size, (self.vocab_size+1)*self.window_size*2))
            vecs = np.zeros((self.vocab_size, self.hidden*2))
            for i, v in enumerate(self.vocab):
                emb = self.word_to_in(v).unsqueeze(0)
                x, h = self.rnn(emb, self.init_hidden(emb))
                x = torch.tanh(x)
                # x = self.lin(x)
                # x = torch.tanh(x)
                # x = self.decoder(x)
                # x = F.log_softmax(x, dim=2)
                vecs[i] = x[0][0]
        return vecs



def evaluate(model, criterion, data):
    with torch.no_grad():
        model.eval()
        total_loss = 0.
        for i, w in enumerate(data):
            if w != '':
                if i % 100 == 0:
                    print('{:f}%'.format(i/(len(data))*100), end='\r')
                emb_out = model.word_to_out(w)
                out = model.forward(w)
                total_loss += criterion(out, emb_out).item()
        return total_loss / len(data)

def train_epoch(model, optimizer, criterion, data):
    model.train()
    shuffle(data)
    for i, word in enumerate(data):
        if word != '':
            optimizer.zero_grad()
            expected = model.word_to_out(word)
            pred = model.forward(word)
            loss = criterion(pred, expected)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('{:f}%, loss: {:f}'.format((i/len(data))*100, loss), end='\r')

def get_rnn_embeddings(data, vocab, window_size, max_epochs, vecspath):
    HIDDEN_SIZE = len(vocab.keys()) * 3

    model = ContextRNN(vocab, HIDDEN_SIZE, window_size)
    loss = nn.NLLLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    prev_eval = np.inf
    eval = 10
    epochs = 0
    best_eval = np.inf
    best_vecs = None

    while epochs < max_epochs:
        prev_eval = eval
        train_epoch(model, optimizer, loss, data)
        eval = evaluate(model, loss, data)
        epochs += 1
        print('Epoch {} eval loss: {}'.format(epochs, eval))
        vecs = model.get_vecs()
        if eval < best_eval:
            best_eval = eval
            best_vecs = vecs
        np.save('{}_{}.npy'.format(vecspath, epochs), vecs)

    return best_vecs
