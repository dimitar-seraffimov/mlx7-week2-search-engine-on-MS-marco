import torch
import model
from pathlib import Path

#
# SETUP
#

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

CHECKPOINT_DIR = Path("./checkpoints")
REG_CKPT_PATH = CHECKPOINT_DIR / "regressor_last.pth"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

#
#
# Find latest CBOW checkpoint
#
#

def get_latest_checkpoint():
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError("Checkpoints directory not found. Run 01_train_w2v.py first.")
    checkpoints = list(CHECKPOINT_DIR.glob('*.cbow.pth'))
    if not checkpoints:
        raise FileNotFoundError("No checkpoint files found in ./checkpoints directory")
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"Loading checkpoint: {latest_checkpoint}")
    return latest_checkpoint


#
#
# Load CBOW model
#
#

cbow = model.CBOW(63642, 128)
cbow.load_state_dict(torch.load(get_latest_checkpoint(), map_location=DEVICE))
cbow = cbow.to(DEVICE)
cbow.eval()

#
#
# Load or initialize regressor
#
#

mReg = model.Regressor().to(DEVICE)
if REG_CKPT_PATH.exists():
    mReg.load_state_dict(torch.load(REG_CKPT_PATH, map_location=DEVICE))

opFoo = torch.optim.Adam(mReg.parameters(), lr=0.005)

#
#
# Dummy input and target
#
#

ipt = torch.tensor([[45, 27, 45367, 456]], device=DEVICE)
trg = torch.tensor([[125.]], device=DEVICE)

#
#
# Train Regressor
#
#

for i in range(100):
    with torch.no_grad():
        emb = cbow.emb(ipt).mean(dim=1)

    out = mReg(emb)
    loss = torch.nn.functional.l1_loss(out, trg)

    loss.backward()
    opFoo.step()
    opFoo.zero_grad()

    if i % 10 == 0:
        print(f"Step {i}: loss = {loss.item():.4f}")

#
#
# Save regressor

torch.save(mReg.state_dict(), REG_CKPT_PATH)
