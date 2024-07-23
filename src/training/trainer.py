import torch
import torch.nn as nn
import torch.optim as optim
from kmspy import DataLoader

from models.diffusion import Diffusion
from training.config import TrainConfig


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.model = Diffusion(config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: max(0.2, 0.98 ** epoch))

    def train_step(self, x: torch.tensor, y: torch.tensor):
        random_t = torch.rand(x.shape[0], device=x.device) * (1-self.config.eps) + self.config.eps
        z = torch.randn_like(x)
        perturbed_x = x + z
        score = self.model(perturbed_x, random_t, y)
        loss = torch.mean(torch.sum((score+z)**2, dim=(1, 2, 3)))
        return loss
    
    def train(self, train_dataloader):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        self.model = self.model.to(device)
        self.model.train()
    
        for e in range(self.config.num_epochs):
            avg_loss = 0.
            num_items = 0

            for step, batch in enumerate(train_dataloader):
                x: torch.tensor = batch["image"].to(device)
                y: torch.tensor = batch["label"].to(device)
                loss = self.train_step(x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]
                if step % 100 == 0:
                    print(f"{e}/{step}: Avg loss: {avg_loss/num_items:.4f}")
            self.scheduler.step()
            lr_current = self.scheduler.get_last_lr()[0]

            print(f"{e}: Avg loss: {avg_loss/num_items:.4f} lr: {lr_current:.3e}")
            torch.save(self.model.state_dict(), 'ckpt_transformer.pth')
