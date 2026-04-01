import torch
from torch import optim
from config import Config
from generator.g_mlp import Generator
from discriminator.d_mlp import Discriminator
from loss.gan_loss import generator_loss, discriminator_loss
from layers.concat_input import concat_input

cfg = Config()

G = Generator(cfg.z_dim, cfg.y_dim, cfg.g_hidden_dim).to(cfg.device)
D = Discriminator(input_dim=784, y_dim=cfg.y_dim, hidden_dim=cfg.d_hidden_dim).to(cfg.device)

g_optimizer = optim.Adam(G.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
for epoch in range(cfg.num_epochs):
    # 1. Sample noise z and label y
    z = torch.randn(64, cfg.z_dim).to(cfg.device)
    y = torch.randint(0, cfg.y_dim, (64,)).to(cfg.device)
    
    # 2. Generate fake data
    fake_data = G(concat_input(z, y))
    
    # 3. Discriminator loss (real data not included, just structure)
    # Placeholder real_data: zeros
    real_data = torch.zeros_like(fake_data).to(cfg.device)
    d_loss = discriminator_loss(D, real_data, fake_data, y)
    
    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()
    
    # 4. Generator loss
    g_loss = generator_loss(D, fake_data, y)
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{cfg.num_epochs}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
