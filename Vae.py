import torch
import torch.nn as nn
from typing import List
from torch import tensor as Tensor
from torchsummary import summary

class VAE_Encoder(nn.Module):
    
    def __init__(self, 
                  in_c: int,
                  latent_dim: int,
                  hidden_dim: List = None,
                  **kwargs) -> Tensor:
        
        super(VAE_Encoder, self).__init__()
        self.in_c = in_c
        self.latent_dim = latent_dim
        if hidden_dim is None:
            hidden_dim = [32,64,128,256,512]
            
        layers = []
        
        for dim in hidden_dim:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels = in_c, out_channels = dim, kernel_size = 3, stride = 2, padding = 1),
                    nn.LeakyReLU(0.2, inplace = True)
                    )
                )
            in_c = dim
            
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        
        return self.encoder(x)

class VAE_Decoder(nn.Module):
    
    def __init__(self, 
                 latent_dim: int,
                 hidden_dim: List = None,
                 **kwargs) -> Tensor:
        
        super(VAE_Decoder, self).__init__()
        self.latent_dim = latent_dim
        if hidden_dim is None:
            hidden_dim = [512,256,128,64,32]
        decoder_layers = []
        in_c = latent_dim
        
        for i, dim in enumerate(hidden_dim):
            
            decoder_layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True),
                    nn.Conv2d(in_channels = in_c, out_channels = dim, kernel_size = 3, padding = 1),
                    nn.ReLU(inplace = True) 
                    )
                )
            in_c = dim
            
        decoder_layers.append(
            nn.Sequential(
                nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True),
                nn.Conv2d(hidden_dim[-1], 3, kernel_size = 3, padding = 1),
                nn.Tanh()
                )
            )
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        
        return self.decoder(x.unsqueeze(-1).unsqueeze(-1))
        

class VAE(nn.Module):
    
    def __init__(self,
                 latent_dim: int,
                 **kwargs) -> None:
        
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.z_shape = 512*4
        self.encoder = VAE_Encoder(3, latent_dim)
        self.fc_mu = nn.Linear(self.z_shape, latent_dim)
        self.fc_sigma = nn.Linear(self.z_shape, latent_dim)
        self.decoder = VAE_Decoder(latent_dim)
        
    def reparameterize(self, mu, var):
        
        sigma = torch.exp(0.5 * var)
        epsilon = torch.randn_like(sigma)
        result = epsilon.mul(sigma).add_(mu)
        return result
    
    def encode(self, x):
        
        z = self.encoder(x)
        z = z.view(-1, self.z_shape)
        mu = self.fc_mu(z)
        var = self.fc_sigma(z)
        z = self.reparameterize(mu, var)
        return z, mu, var

    def forward(self, x):

        z, mu, var = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, mu, var        
        
model = VAE(128)
print(summary(model, (3,64,64), device = 'cpu'))     