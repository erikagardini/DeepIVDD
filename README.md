# DeepIVDD

## Pretrain the networks
### ContrDA
Download the project from https://github.com/google-research/deep_representation_one_class
Run the training step and save the resulting models in the folder "python/nets/contrDA/contrDA_pretraining"

### SimSiam
Run the training step using the file "python/nets/simsiam/simsiam_pretraining"

### VAE
Run the training step using the file "python/nets/vae/vae_pretraining"

# DeepIVDD
Run the experiments using the files: "python/main_ssl.py", "python/main_simsiam.py", "python/main_vae.py"