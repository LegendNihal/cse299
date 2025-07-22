import torchvision.utils as vutils
from models import get_model
from datasets import get_dataset
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import random
from tqdm import tqdm

def run_pipeline(model_name, dataset_name, options_name=None):
    # This will run in main.py
    if model_name == "dcgan":
        gan = get_model(model_name)
        dataloader = get_dataset(dataset_name)
        gan.train(dataloader, num_epochs=5)
        samples = gan.generate_samples(16)
        results_dir = os.path.join("RESULTS", "dcgan")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "dcgan_generated_samples.png")
        vutils.save_image(samples, save_path, normalize=True)
    elif model_name == "cgan":
        cgan = get_model(model_name)
        dataloader = get_dataset(dataset_name,model_name)
        cgan.train(dataloader, epochs=50, save_interval=10)
        gen_imgs, labels = cgan.generate_samples(num_samples=3, specific_class=7)
        results_dir = os.path.join("RESULTS", "cgan")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "cgan_generated_samples.png")
        vutils.save_image(gen_imgs, save_path, normalize=True)
    elif model_name == "cvae":
        cvae = get_model(model_name)
        dataloader = get_dataset(dataset_name,model_name)
        cvae.train(dataloader, epochs=50, save_interval=10)
        gen_imgs, labels = cvae.generate_samples(num_samples=10, specific_class=4)
        results_dir = os.path.join("RESULTS", "cvae")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "cvae_generated_samples.png")
        vutils.save_image(gen_imgs, save_path, normalize=True)
        cvae.generate_class_samples(save_path = os.path.join(results_dir, "cvae_class_samples.png"))
    elif model_name == "wavegan":
        LATENT_DIM = 100
        AUDIO_LENGTH = 16384
        NUM_EPOCHS = 50
        LEARNING_RATE = 0.0001
        generator, discriminator = get_model(model_name)
        dataset_dir = get_dataset(dataset_name)
        class AudioDataset(Dataset):
            """Custom dataset for loading audio files"""
            
            def __init__(self, audio_dir, sample_rate=16000, audio_length=16384):
                self.audio_dir = Path(audio_dir)
                self.sample_rate = sample_rate
                self.audio_length = audio_length
                
                # Get all audio files
                self.audio_files = []
                for ext in ['*.wav', '*.mp3', '*.flac']:
                    self.audio_files.extend(self.audio_dir.glob(f"**/{ext}"))
                
                print(f"Found {len(self.audio_files)} audio files")
                
            def __len__(self):
                return len(self.audio_files)
            
            def __getitem__(self, idx):
                audio_path = self.audio_files[idx]
                
                try:
                    # Load audio using torchaudio
                    waveform, sr = torchaudio.load(audio_path)
                    
                    # Convert to mono if stereo
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    # Resample if necessary
                    if sr != self.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                        waveform = resampler(waveform)
                    
                    # Ensure we have the right length
                    if waveform.shape[1] > self.audio_length:
                        # Random crop
                        start = random.randint(0, waveform.shape[1] - self.audio_length)
                        waveform = waveform[:, start:start + self.audio_length]
                    elif waveform.shape[1] < self.audio_length:
                        # Pad with zeros
                        pad_length = self.audio_length - waveform.shape[1]
                        waveform = F.pad(waveform, (0, pad_length))
                    
                    # Normalize to [-1, 1]
                    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
                    
                    return waveform.squeeze(0)  # Remove channel dimension
                    
                except Exception as e:
                    print(f"Error loading {audio_path}: {e}")
                    # Return silence if file can't be loaded
                    return torch.zeros(self.audio_length)
        dataset = AudioDataset(dataset_dir, audio_length=16384)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
        test_batch_size = 2
        test_noise = torch.randn(test_batch_size, LATENT_DIM).to(torch.device)
        test_audio = torch.randn(test_batch_size, AUDIO_LENGTH).to(torch.device)
        try:
            gen_output = generator(test_noise)
            print(f"Generator output shape: {gen_output.shape}")
            
            disc_output = discriminator(test_audio)
            print(f"Discriminator output shape: {disc_output.shape}")
            
            disc_gen_output = discriminator(gen_output)
            print(f"Discriminator on generated audio shape: {disc_gen_output.shape}")
            
            print("✓ Model architectures are working correctly!")
            
        except Exception as e:
            print(f"✗ Error in model architecture: {e}")
            return
        def train_wavegan(generator, discriminator, dataloader, num_epochs=100, lr=0.0001):
            """Train WaveGAN"""
            
            criterion = nn.BCEWithLogitsLoss()
            
            optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
            optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
            
            generator.train()
            discriminator.train()

            fixed_noise = torch.randn(16, generator.latent_dim).to(torch.device)

            G_losses = []
            D_losses = []
            
            for epoch in range(num_epochs):
                epoch_G_loss = 0
                epoch_D_loss = 0
                
                progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
                
                for i, real_audio in enumerate(progress_bar):
                    batch_size = real_audio.size(0)
                    real_audio = real_audio.to(torch.device)
                    
                    real_labels = torch.ones(batch_size, 1).to(torch.device)
                    fake_labels = torch.zeros(batch_size, 1).to(torch.device)
                    
                    optimizer_D.zero_grad()
                    
                    output_real = discriminator(real_audio)
                    d_loss_real = criterion(output_real, real_labels)

                    noise = torch.randn(batch_size, generator.latent_dim).to(torch.device)
                    fake_audio = generator(noise)
                    output_fake = discriminator(fake_audio.detach())
                    d_loss_fake = criterion(output_fake, fake_labels)
                    
                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward()
                    optimizer_D.step()
                    
                    optimizer_G.zero_grad()
                    
                    output_fake = discriminator(fake_audio)
                    g_loss = criterion(output_fake, real_labels)
                    
                    g_loss.backward()
                    optimizer_G.step()
                    
                    epoch_G_loss += g_loss.item()
                    epoch_D_loss += d_loss.item()
                    
                    progress_bar.set_postfix({
                        'G_loss': f'{g_loss.item():.4f}',
                        'D_loss': f'{d_loss.item():.4f}'
                    })
                
                avg_G_loss = epoch_G_loss / len(dataloader)
                avg_D_loss = epoch_D_loss / len(dataloader)
                
                G_losses.append(avg_G_loss)
                D_losses.append(avg_D_loss)
                
                print(f'Epoch [{epoch+1}/{num_epochs}] - G_loss: {avg_G_loss:.4f}, D_loss: {avg_D_loss:.4f}')
                
                if (epoch + 1) % 10 == 0:
                    generator.eval()
                    with torch.no_grad():
                        fake_samples = generator(fixed_noise)
                        sample_audio = fake_samples[0].cpu().numpy()
                        
                        os.makedirs('RESULTS/wavegan', exist_ok=True)
                        
                        torchaudio.save(
                            f'RESULTS/wavegan/train_samples/sample_epoch_{epoch+1}.wav',
                            torch.tensor(sample_audio).unsqueeze(0),
                            16000
                        )
                    generator.train()
            
            return G_losses, D_losses

        def generate_samples(generator, num_samples=5):
            """Generate audio samples"""
            generator.eval()
            
            os.makedirs('final_samples', exist_ok=True)
            
            with torch.no_grad():
                for i in range(num_samples):
                    noise = torch.randn(5, generator.latent_dim).to(torch.device)
                    fake_audio = generator(noise)
                    
                    audio_np = fake_audio[0].cpu().numpy()
                    
                    torchaudio.save(
                        f'RESULTS/wavegan/final_samples/generated_sample_{i+1}.wav',
                        torch.tensor(audio_np).unsqueeze(0),
                        16000
                    )
                    
                    print(f"Generated sample {i+1} saved")
        print(f"\nStarting training for {NUM_EPOCHS} epochs...")
        train_wavegan(generator, discriminator, dataloader,num_epochs=NUM_EPOCHS, lr=LEARNING_RATE)
        print("\nGenerating final samples...")
        generate_samples(generator, num_samples=5)

    else:
        return
