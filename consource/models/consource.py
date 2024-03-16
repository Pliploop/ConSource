import torch
import pytorch_lightning as pl
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
from audio_diffusion_pytorch.diffusion import *
from pytorch_lightning.cli import OptimizerCallable

class CustomVDiffusion(VDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:  # type: ignore
        batch_size, device = x.shape[0], x.device
        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        # Get noise
        if noise is None:
            noise = torch.randn_like(x)
        else:
            noise = noise.to(device)
        # Combine input and noise weighted by half-circle
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x
        # Predict velocity and return loss
        v_pred = self.net(x_noisy, sigmas, **kwargs)
        return self.loss_fn(v_pred, v_target)

class ConSource(pl.LightningModule):
    def __init__(self, encoder, encoder_ckpt = None, freeze_encoder = True, cfg_training=0.1, embedding_max_length = 1,optimizer: OptimizerCallable = None):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.diffuser = DiffusionModel(
            net_t = UNetV0,
            in_channels = 1, # U-Net: number of input channels
            channels=[256, 512,1024,1024,1024,1024], # U-Net: channels at each layer
            factors=[4, 4,4,4,4,4], # U-Net: downsampling and upsampling factors at each layer
            items=[2, 2, 2, 2, 2, 2], # U-Net: number of repeating items at each layer
            attentions=[0, 0,0,1,1,1], # U-Net: attention enabled/disabled at each layer
            attention_heads=8, # U-Net: number of attention heads per attention item
            attention_features=64, # U-Net: number of attention features per attention item
            diffusion_t=VDiffusion, # The diffusion method used
            sampler_t=VSampler, # The diffusion sampler used
            use_text_conditioning=False,
            use_embedding_cfg = False,
            embedding_max_length = embedding_max_length,
            embedding_features = 512
        )
        
        self.optimizer = optimizer
        self.cfg_training = cfg_training # classifier free guidance. 1: no guidance, 0: full guidance
        
        if encoder_ckpt is not None:
            self.encoder.load_state_dict(torch.load(encoder_ckpt))
            print(f"Encoder loaded from {encoder_ckpt}")
        
        if freeze_encoder:
            self.encoder.freeze()
            self.encoder.eval()
            print("Encoder frozen")
    
    def extract_avg_embedding(self, conditioning):
        in_samples = self.encoder.in_samples
        # conditioning is of shape [batch,1,t]
        # split into chunks of in_samples and pad the last one
        chunks = torch.split(conditioning, in_samples, dim = 2)
        last_chunk = chunks[-1]
        last_chunk = torch.nn.functional.pad(last_chunk, (0, in_samples - last_chunk.shape[2]))
        chunks = chunks[:-1] + (last_chunk,)
        chunks = torch.stack(chunks, dim = 1)
        batch_embeddings = [
            self.encoder(chunk) for chunk in chunks
        ]
        batch_embeddings = torch.stack(batch_embeddings, dim = 0)
        return batch_embeddings
    
    def training_step(self, x):
        target = x['target']
        noise = x['accomp']
        
        conditioning = self.extract_avg_embedding(target)
        loss = self.diffuser(target, noise, noise = noise, embedding_features = conditioning, embedding_mask_proba = self.cfg_training)
        
        return loss
    
    
    def forward(self, x, num_steps= 10, *args, **kwargs):
        
        target = x['target']
        noise = x['mix']
        with torch.no_grad():
            
            conditioning = self.extract_avg_embedding(target)
            output = self.diffuser.sample(noise, num_steps = num_steps, embedding = conditioning, *args, **kwargs)
        
        return output
        
    def configure_optimizers(self):
        if self.optimizer is None:
            return torch.optim.Adam(self.parameters(), lr=1e-4)
        return self.optimizer(self.parameters())