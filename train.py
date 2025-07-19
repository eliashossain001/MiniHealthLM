import torch
from torch.utils.data import DataLoader
from src.config import Config
from src.dataset import TextDataset
from src.model import MiniHealthLM
from src.utils import save_checkpoint
from tqdm import tqdm

import os
os.makedirs("checkpoints", exist_ok=True)

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniHealthLM(config).to(device)

dataset = TextDataset("data/corpus.txt", config)
loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

step = 0
model.train()
for epoch in range(config.epochs):
    print(f"\n🟢 Starting Epoch {epoch + 1}/{config.epochs}")
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", dynamic_ncols=True)

    for batch in pbar:
        batch = batch.to(device)
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        # Live update in progress bar
        pbar.set_postfix(step=step, loss=loss.item())

        # Save checkpoint
        if step % config.save_every == 0:
            save_checkpoint(model, optimizer, step, path=f"checkpoints/ckpt_{step}.pt")
            print(f"💾 Saved checkpoint at step {step}")

        if step >= config.total_steps:
            print(f"✅ Reached total steps ({config.total_steps}). Stopping training.")
            break
