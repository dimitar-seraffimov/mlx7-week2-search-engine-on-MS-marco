import torch
import pickle

#
#
#
#
#

class Vocabulary(torch.utils.data.Dataset):
  def __init__(self):
    self.vocab_to_int = pickle.load(open('./tkn_vocab_to_int.pkl', 'rb'))
    self.int_to_vocab = pickle.load(open('./tkn_int_to_vocab.pkl', 'rb'))
    self.corpus = pickle.load(open('./combined_corpus.pkl', 'rb'))
    self.tokens = [self.vocab_to_int.get(word, self.vocab_to_int["<UNK>"]) for word in self.corpus]

  def __len__(self):
    return len(self.tokens)

  def __getitem__(self, idx: int):
    ipt = self.tokens[idx]
    prv = self.tokens[idx-2:idx]
    nex = self.tokens[idx+1:idx+3]
    if len(prv) < 2: prv = [0] * (2 - len(prv)) + prv
    if len(nex) < 2: nex = nex + [0] * (2 - len(nex))
    return torch.tensor(prv + nex), torch.tensor([ipt])


#
#
#
if __name__ == '__main__':
  ds = Vocabulary()
  print(ds.tokens[:15])
  # print(ds[0])
  print(ds[5])
  dl = torch.utils.data.DataLoader(dataset=ds, batch_size=3)
  ex = next(iter(dl))
  print(ex)