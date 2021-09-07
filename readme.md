## Variational AutoEncoder (VAE) 

- VAEs compresses higher dimensional data to lower dimensional bottleneck representation.
- The goal of VAE is to find a distribution q(z|x) of some latent variables which we can sample from z (bottleneck) to generate new samples x' ~ p(x|z). 
- Differences between vanilla AutoEncoders and VAEs are:
1) AutoEncoders can only reconstruct the data-points that are present in the dataset. If we pass a random datapoint which is not from a dataset distribution, Decoder of AutoEncoder will generate random garbage image. While, VAEs can generate images from various datapoints in the distribution.
2) Loss function of Autoencoder only consists of reconstruction loss(MSE loss). While, VAE loss function consists of KL-Divergence loss along with MSE loss. 

### Results
- Epoch 1 vs Epoch 25
<p float="left" align = 'center'>
  <img src="https://github.com/Vrushank264/VAE-PyTorch/blob/main/Generated%20Images/img_01.png" width = 400/>
  :arrow_right:
  <img src="https://github.com/Vrushank264/VAE-PyTorch/blob/main/Generated%20Images/img_03.png" width = 400/> 
</p>
- You can see more examples in `Generated Images` directory.

### Details
- This project uses CelebA Dataset.</br>
- VAE is trained on 64x64 image patches for 25 epochs.</br>
- Trained model is provided in `Trained model` directory.</br>
