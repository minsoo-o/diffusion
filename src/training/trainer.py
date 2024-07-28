import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.diffusion import Diffusion
from training.config import TrainConfig


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.marginal_prob_std = self.set_marginal_prob_std(self.config.sigma)
        self.diffusion_coeff = self.set_diffusion_coeff(self.config.sigma)

        self.model = Diffusion(config, marginal_prob_std=self.marginal_prob_std)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: max(0.01, 0.98 ** epoch))

        if not os.path.exists(self.config.savedir):
            os.makedirs(self.config.savedir)

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.model = self.model.to(self.device)
    
    def set_marginal_prob_std(self, sigma):
        def _wrapper(t):
            return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))
        return _wrapper
    
    def set_diffusion_coeff(self, sigma):
        def _wrapper(t):
            return sigma**t
        return _wrapper
    
    def train_step(self, x: torch.tensor, y: torch.tensor):
        random_t = torch.rand(x.shape[0], device=x.device) * (1-self.config.eps) + self.config.eps
        z = torch.randn_like(x)
        std = self.marginal_prob_std(random_t)
        perturbed_x = x + z*std[:, None, None, None]
        score = self.model(perturbed_x, random_t, y)
        loss = torch.mean(torch.sum((score*std[:, None, None, None] + z)**2, dim=(1, 2, 3)))
        return loss
    
    def sample(self, x_size, device):
        """
        Euler–Maruyama method
        """
        batch_size = self.config.n_classes

        # target label
        labels = torch.arange(0, self.config.n_classes, step=1).to(device)

        # initial x
        images = torch.randn((batch_size, *x_size)).to(device) * self.marginal_prob_std(torch.ones(batch_size, device=device))[:, None, None, None]

        # time steps
        time_steps = torch.linspace(1., self.config.eps, self.config.time_steps, device=device)
        dt = time_steps[0] - time_steps[1]

        self.model.eval()
        with torch.no_grad():
            for time_step in time_steps:
                batch_time_step = time_step[None]
                g = self.diffusion_coeff(batch_time_step)
                mean_images = images + (g**2)[:, None, None, None] * self.model(images, batch_time_step, labels)*dt
                images = mean_images + torch.sqrt(dt) * g[:, None, None, None] * torch.randn_like(images)
        return mean_images, labels
    
    def save_plot(self, images, labels, epoch):
        cols, rows = 5, self.config.n_classes//5 + (1 if self.config.n_classes%5 != 0 else 0)
        fig, axs = plt.subplots(rows, cols, figsize=(15, 3 * rows))

        for i in range(self.config.n_classes):
            row = i//cols
            col = i%cols
            ax = axs[row, col] if rows > 1 else axs[col]
            img = images[i].permute(1, 2, 0)
            ax.imshow(img.detach().cpu(), cmap=cm.binary)
            ax.set_title(labels[i].item())
            ax.axis("off")
        plt.savefig(f"{self.config.savedir}/e{epoch}_sample.png", dpi=300)
        plt.close()

    def train(self, train_dataloader):
        device = self.device

        self.model = self.model.to(device)
        self.model.train()

        step = 0
        for e in range(self.config.num_epochs):
            avg_loss = 0.
            num_items = 0

            for batch in train_dataloader:
                step += 1
                x: torch.tensor = batch["image"].to(device)
                y: torch.tensor = batch["label"].to(device)
                loss = self.train_step(x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]
                # if step % 100 == 0:
                #     print(f"{e}/{step}: Avg loss: {avg_loss/num_items:.4f}")
            self.scheduler.step()
            lr_current = self.scheduler.get_last_lr()[0]

            torch.save(self.model.state_dict(), f'{self.config.savedir}/ckpt_transformer.pth')
            print(f"{e}: Avg loss: {avg_loss/num_items:.4f} lr: {lr_current:.3e}")
            images, labels = self.sample(x_size=list(x.size())[1:], device=device)
            self.save_plot(images, labels, e)
