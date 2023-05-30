import torch
from torch.utils.data import DataLoader
import os


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        loss_fn: torch.nn.modules.loss,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        # self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path: str):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_fn(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch: int):
        batch_size = len(next(iter(self.dataloader))[0])
        print(f"\n[GPU{self.global_rank}] Epoch {epoch} | Batch size: {batch_size} | Steps: {len(self.dataloader)}")
        # self.dataloader.sampler.set_epoch(epoch)
        for source, targets in self.dataloader:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch: int):
        snapshot = {
            "MODEL_STATE": self.model.state_dict(), # self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
