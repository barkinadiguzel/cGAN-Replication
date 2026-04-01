import torch
import torch.nn as nn

bce_loss = nn.BCELoss()

def generator_loss(D, fake_data, y):
    pred = D(fake_data)
    target = torch.ones_like(pred)
    return bce_loss(pred, target)

def discriminator_loss(D, real_data, fake_data, y):
    pred_real = D(real_data)
    pred_fake = D(fake_data.detach())
    
    real_target = torch.ones_like(pred_real)
    fake_target = torch.zeros_like(pred_fake)
    
    loss_real = bce_loss(pred_real, real_target)
    loss_fake = bce_loss(pred_fake, fake_target)
    
    return loss_real + loss_fake
