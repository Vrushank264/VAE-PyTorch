import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from Vae import VAE

batch_size = 64
num_epochs = 100
latent_dim = 128
MODEL_PATH = 'E:/Computer Vision/VAE/vae.pth'
GIF_PATH = 'E:/Computer Vision/VAE/'

def vae_loss(imgs, outputs, mu, var):
    
    recons_loss = nn.MSELoss(reduction='sum')(imgs, outputs)
    kl_divergence_loss  = -0.5 * torch.sum(var + 1 - mu**2 - torch.exp(var))
    return kl_divergence_loss + recons_loss

def gif(img_list, path):
    
    fig = plt.figure(figsize=(8,8))
    plt.axis('off')
    ims = [[plt.imshow(np.transpose(i,(1,2,0)),animated = True)] for i in img_list]
    animation = anim.ArtistAnimation(fig,ims,interval = 100,repeat_delay = 100, blit = True)
    plt.show()
    animation.save(path,dpi = 100, writer='imagekclick')
    
SetRange = T.Lambda(lambda X: 2 * X - 1.)
transform = T.Compose(
    [
     T.Resize(64,64),
     T.ToTensor(),
     SetRange
     ])

train_data = datasets.CelebA(root = '/content/train', split = 'train', transform = transform, download = True)
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 4)

def train(loader, latent_dim, batch_size, num_epochs, device = torch.device('cuda')):
    
    model = VAE(128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
    img_list = []
    
    for epoch in range(1, num_epochs + 1):
        
        run_loss = 0.0
        model.train()
        
        for i, (imgs, _) in enumerate(loader):
            
            imgs = imgs.to(device)
            model.zero_grad()
            outputs, mu, var = model(imgs)
            loss = vae_loss(imgs, outputs, mu, var)
            run_loss += loss.item()
            loss.backward()
            opt.step()
            
            if i % 250 == 0:
                
                with torch.no_grad():
                    gen_imgs = model(128).detach().cpu()
                img_list.append(vutils.make_grid(gen_imgs, normalize = True))
                plt.axis('off')
                plt.imshow(np.transpose(img_list[-1] ,(1,2,0)))
                plt.show()
                
        run_loss /= len(loader)
        print("Loss at epoch {}: {}".format(epoch, torch.mean(run_loss)))
        torch.save(model.state_dict(), path = MODEL_PATH)
    
    gif(img_list, GIF_PATH)
        
if __name__ == '__main__':
    
    train(train_loader, latent_dim, batch_size, num_epochs)