<h1> PyTorch Implement of Variational AutoEncoder (VAE) </h1>

- VAEs compresses higher dimensional data to lower dimensional bottleneck representation.
- The goal of VAE is to find a distribution q(z|x) of some latent variables which we can sample from z (bottleneck) to generate new samples x' ~ p(x|z). 
- Differences between vanilla AutoEncoders and VAEs are:
1) AutoEncoders can only reconstruct the data-points that are present in the dataset. If we pass a random datapoint which is not from a dataset distribution, Decoder of AutoEncoder will generate random garbage image. While, VAEs can generate images from various datapoints in the distribution.
2) Loss function of Autoencoder only consists of reconstruction loss(MSE loss). While, VAE loss function consists of KL-Divergence loss along with MSE loss. 

<h3>Results</h3>

<p float="left" align = 'center'>
  <img src="" width = 256/>
  :arrow_right:
  <img src=""/> 
</p>


<h3> Details </h3>
- This project uses CelebA Dataset.</br>
- VAE is trained on 64x64 image patches for 25 epochs.</br>
- Trained model is provided in `Trained model` directory.</br>
