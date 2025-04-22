import torch

#
#
#
#
#

class CBOW(torch.nn.Module):
  def __init__(self, voc, emb):
      super().__init__()
      self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
      self.ffw = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)

  def forward(self, inpt):
      emb = self.emb(inpt)
      emb = emb.mean(dim=1)
      return self.ffw(emb)

#
#
#
#
#

class Regressor(torch.nn.Module):
  def __init__(self):
      super().__init__()
      self.seq = torch.nn.Sequential(
          torch.nn.Linear(128, 64),
          torch.nn.ReLU(),
          torch.nn.Dropout(0.2),

          torch.nn.Linear(64, 32),
          torch.nn.ReLU(),
          torch.nn.Dropout(0.2),

          torch.nn.Linear(32, 16),
          torch.nn.ReLU(),
          torch.nn.Dropout(0.2),

          torch.nn.Linear(16, 1),
      )

  def forward(self, inpt):
      return self.seq(inpt)

#
#
#
#
#

if __name__ == "__main__":
  model = CBOW(128, 8)
  print("CBOW:", model)
  criterion = torch.nn.CrossEntropyLoss()
  inpt = torch.randint(0, 128, (3, 5))
  trgt = torch.randint(0, 128, (3,))
  out = model(inpt)
  loss = criterion(out, trgt)
  print(loss)