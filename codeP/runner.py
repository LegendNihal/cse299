import torchvision.utils as vutils
from models import get_model
from datasets import get_dataset
import os

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
    else:
        return
