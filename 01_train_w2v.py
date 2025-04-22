import tqdm
import wandb
import torch
import datetime
import model
import dataset
import evaluate
from pathlib import Path


#
# SETUP
#
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

# Create checkpoints directory
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


#
# LOAD DATA
#
ds = dataset.Vocabulary()
dl = torch.utils.data.DataLoader(dataset=ds, batch_size=256, shuffle=True)

#
# INIT MODEL
#

vocab_size = len(ds.int_to_vocab)
embedding_dim = 128

model_cbow = model.CBOW(vocab_size, embedding_dim)
print('Model parameters:', sum(p.numel() for p in model_cbow.parameters()))

optimizer = torch.optim.Adam(model_cbow.parameters(), lr=0.003)
criterion = torch.nn.CrossEntropyLoss()

#
# Initialize wandb
#
wandb.init(
    project='mlx7-week2-search-engine',  # Your project name
    name=f'cbow_{timestamp}'
)
model_cbow.to(device)

#
# TRAIN MODEL
#
for epoch in range(15):
    progress_bar = tqdm.tqdm(dl, desc=f"Epoch {epoch + 1}", leave=False)
    for i, (context, target) in enumerate(progress_bar):
        context, target = context.to(device), target.to(device)

        optimizer.zero_grad()
        output = model_cbow(context)
        loss = criterion(output, target.squeeze())
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss.item()})

        if i % 10000 == 0:
            evaluate.topk(model_cbow)

    # Save checkpoint
    ckpt_path = CHECKPOINT_DIR / f"{timestamp}.{epoch + 1}.cbow.pth"
    torch.save(model_cbow.state_dict(), ckpt_path)

    artifact = wandb.Artifact("model-weights", type="model")
    artifact.add_file(str(ckpt_path))  # Convert to string for wandb
    wandb.log_artifact(artifact)

#
#
#
#
#
wandb.finish()