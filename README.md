# cse299
README:  Unified Generative AI Models via Command-Line Interface 
This repository provides a unified framework for running multiple generative AI models via a Terminal CLI. The framework allows you to fine-tune different models for tasks such as text generation, image generation, and audio synthesis, all from a simple command line interface.

 
### Supported Models
 
The project includes implementations for the following models:
 
- **GPT-2**: A transformer-based model for text generation.
- **WaveGAN**: A generative adversarial network for raw audio waveform generation.
- **CGAN (Conditional GAN)**: A GAN model where the generation is conditioned on label input.
- **DCGAN (Deep Convolutional GAN)**: A convolutional GAN model used primarily for image generation.
- **CVAE (Conditional Variational Autoencoder)**: A model that learns to generate data conditioned on certain attributes using variational inference.
 
---
 
## Directory Structure
CSE299/
 ├── .vscode/ # VSCode workspace settings (optional)
 ├── codeB/ # Bash scripts to run models
 │ ├── cgan.sh
 │ ├── cvae.sh
 │ ├── dcgan.sh
 │ ├── gpt2.sh
 │ └── wavegan.sh
 ├── codeP/ # Python source code
 │ ├── datasets/ # Custom dataset loaders for each model
 │ │ ├── cgan_dataset.py
 │ │ ├── cvae_dataset.py
 │ │ ├── dcgan_dataset.py
 │ │ ├── gpt2_dataset.py
 │ │ └── wavegan_dataset.py
 │ ├── models/ # Model implementations
 │ │ ├── cgan.py
 │ │ ├── cvae.py
 │ │ ├── dcgan.py
 │ │ ├── gpt2.py
 │ │ └── wavegan.py
 │ ├── main.py # Entry point for running training or evaluation
 │ └── runner.py # Handles training loops and logging
 ├── DATA/ # Dataset storage
 │ └── dataset_name/
 │ ├── training/ # Training data
 │ └── testing/ # Testing data
 ├── RESULTS/ # Output directory for saved results, logs, and models
 ├── others/ # Miscellaneous files or scripts
 ├── .gitignore # Files to ignore in version control
 ├── LICENSE # Project license file
 ├── README.md # Project documentation
 └── requirements.txt # Python dependencies
 
---
 
## How to Run a Model
 
You can run any of the models using the main Python file. Example usage:
 
```bash
python codeP/main.py --dataset dataset_name --model model_name --otherOptions other_options
bash codeB/gpt2.sh

GPT-2 Fine-Tuning Module
The GPT-2 module in this project provides a complete pipeline for fine-tuning a pre-trained GPT-2 model on custom text data. It supports data ingestion from .txt, .pdf, and .docx formats, making it flexible for various domains like reports, articles, or transcripts.

1. Dataset Preparation
Your training data should be placed inside the directory:
DATA/training/

This directory can contain any number of the following supported file types:
.txt – Plain text files


.docx – Microsoft Word documents


.pdf – Portable Document Format files


When training begins, these files will be:
Read and concatenated


Cleaned and de-duplicated


Saved into a single file named processed_training_data.txt inside the same training/ folder


This file is then tokenized and fed into the GPT-2 model for fine-tuning.

2. Training the Model
Training is managed by the GPT2FineTuner class defined in codeP/models/gpt2.py.
To train the model:
python ./codeP/main.py --dataset text_data --model gpt2 --otherOptions default

This command will:
Load and preprocess training data from DATA/training/


Initialize the GPT-2 tokenizer and model


Fine-tune the model using Hugging Face's Trainer API


Save the model and tokenizer inside ./DATA/output/


The model will be trained using default hyperparameters: batch size = 2, epochs = 30, and block size = 256 tokens. You can change these in GPT2FineTuner.
Alternatively, run the provided shell script:
bash codeB/gpt2.sh

This script wraps the above command for convenience.

3. Text Generation
After training is complete, you can generate new text using the fine-tuned model. The model supports:
Generating text from a single prompt


Reading multiple prompts from a file and saving the results


Prompts should be stored in a text file (e.g., prompts.txt) with one prompt per line. Comment lines (starting with #) are ignored.
Example output includes:
Individual result files: result_001.txt, result_002.txt, etc.


A combined result file: all_results_YYYYMMDD_HHMMSS.txt


These will be saved under a directory you specify (e.g., RESULTS/gpt2_output/).

4. Output Directory Structure
After training and generation, outputs are saved in:
DATA/
└── output/                     # Trained GPT-2 model and tokenizer

RESULTS/
└── gpt2_output/                # Generated text outputs from prompts
    ├── result_001.txt
    ├── result_002.txt
    └── all_results_20250805_1530.txt


5. Logging and Progress
The script provides clear logging throughout:
Device being used (CPU or GPU)


Number of training files found


Prompt processing progress


Any errors encountered



Conditional GAN (CGAN) Module
The CGAN module in this project allows you to train a Conditional Generative Adversarial Network on labeled datasets. This architecture generates images conditioned on class labels, making it suitable for tasks where control over the output category is desired (e.g., generating digits 0–9 from MNIST).

1. Dataset Used
This implementation uses the MNIST dataset, which consists of grayscale handwritten digit images (28x28 pixels) and their corresponding labels (0–9).
The dataset is automatically downloaded and saved to:
DATA/cgan_dataset/training/data/

You do not need to manually download or prepare MNIST. The cgan_dataset.py script handles everything.
Each image is normalized to a pixel range of [-1, 1] for compatibility with the generator's Tanh activation output.

2. How the CGAN Works
The CGAN architecture consists of:
Generator: Takes a random noise vector and a class label as input and produces an image that resembles samples from that class.


Discriminator: Takes an image and a class label as input and determines whether the image is real and correctly conditioned.


Both networks use nn.Embedding to condition the model on class labels and are trained in an adversarial manner.

3. Training the CGAN
You can initiate training using the following command:
python ./codeP/main.py --dataset mnist --model cgan --otherOptions default

Alternatively, you can run the provided shell script:
bash codeB/cgan.sh

This will:
Load the MNIST dataset via torchvision


Initialize the CGAN architecture (Generator and Discriminator)


Train both networks for the number of epochs defined in the code


Log training progress (losses for both Generator and Discriminator)


Each epoch logs average loss values and progress using a progress bar (powered by tqdm).
4. Output and Sample Generation
After training, you can generate and visualize output samples:
Random Samples: The method generate_samples() creates images for random classes.


Per-Class Samples: The method generate_class_samples() generates 5 images per class (0–9) and saves a combined image grid to:


class_samples.png

This is useful for visualizing how well the generator has learned to conditionally produce images for each class.

5. Output Directory Structure
Example output layout:
DATA/
└── cgan_dataset/
    └── training/
        └── data/          # Contains downloaded MNIST images

RESULTS/
└── cgan_output/            # (Optional) Save plots, checkpoints, etc.

class_samples.png           # Grid of generated images across all classes

You can modify the save path or integrate result saving with the runner.py logic if needed.
6. Hyperparameters (Default Values)
The CGAN module uses the following default training settings:
Parameter
Value
Noise Dimension
100
Batch Size
64
Image Size
28×28
Learning Rate
0.0002
Optimizer
Adam
Epochs
100
Label Embedding
10 classes

These can be adjusted by editing the ConditionalGAN class in models/cgan.py.
7. Notes
The model uses .to(self.device) to automatically switch between CPU and GPU if CUDA is available.


Loss functions use Binary Cross-Entropy Loss (BCELoss) for adversarial training.


Dropout layers are included in the Discriminator to reduce overfitting.



Conditional VAE (CVAE) Module
The Conditional Variational Autoencoder (CVAE) module allows training a generative model that learns to reconstruct and generate images conditioned on class labels. Unlike GANs, CVAEs learn a probabilistic latent space and are trained using reconstruction and regularization losses.

1. Overview
The CVAE architecture consists of:
Encoder: Takes an input image and a label, and outputs the parameters (mean and log-variance) of a latent Gaussian distribution.


Decoder: Takes a sampled latent vector and a label, and reconstructs the original image.


Latent Sampling: The model uses the reparameterization trick to sample from the latent distribution during training.


This architecture is particularly useful for learning interpretable latent spaces and generating new samples in a structured manner.

2. Dataset
The model is trained on the MNIST dataset, which consists of grayscale images (28×28 pixels) of handwritten digits (0–9).
The dataset is automatically downloaded to:
DATA/cvae_dataset/training/data/

The images are preprocessed using standard normalization and tensor conversion. No manual intervention is needed.

3. Training the CVAE
To train the model, use:
python ./codeP/main.py --dataset mnist --model cvae --otherOptions default

Or run the corresponding bash script:
bash codeB/cvae.sh

This command will:
Download and load the MNIST dataset


Initialize the encoder and decoder


Train the model for the defined number of epochs (default: 100)


Save sample outputs and reconstructions at regular intervals


Training progress is shown via tqdm with loss metrics:
Total Loss


Reconstruction Loss


KL Divergence Loss


These metrics are also stored for plotting if needed.

4. Output and Visualization
The model supports various output visualizations:
a. Sample Generation
The model can generate synthetic images conditioned on specific labels using generate_samples(). This enables controlled generation of digits (e.g., only class = 7).
b. Per-Class Samples
generate_class_samples(save_path='RESULTS/cvae/class_samples.png')

Generates 5 images per class and saves them as a grid image.
c. Image Reconstruction
During training, the model saves comparisons between original and reconstructed images:
RESULTS/cvae/cvae_samples/reconstruction_epoch_10.png

This helps evaluate how well the model captures the underlying data distribution.
d. Latent Space Interpolation
The model supports interpolation between two classes in the latent space:
interpolate_in_latent_space(class1=2, class2=8, num_steps=10)

This visually shows smooth transitions from one class to another.

5. Output Directory Structure
After training, the following structure will be populated:
RESULTS/
└── cvae/
    └── cvae_samples/
        ├── generated_epoch_10.png            # Generated sample grid
        ├── reconstruction_epoch_10.png       # Reconstruction comparison
        └── ...                               # Other saved epochs


6. Default Hyperparameters
Parameter
Value
Latent Dimension
20
Image Size
28×28
Batch Size
64
Learning Rate
0.001
Optimizer
Adam
Epochs
100
Beta (KL weight)
1.0

These values can be modified by editing the ConditionalVAE class in models/cvae.py.

7. Notes
The model is compatible with GPU acceleration.


Outputs are saved periodically during training.


You can use the plot_losses() function to visualize training progress.



Here's the next section of your README.md that explains the WaveGAN module in a clean, professional, and beginner-friendly format, following the same documentation style used for GPT-2, CGAN, and CVAE.

WaveGAN Module
The WaveGAN module provides an implementation of a Generative Adversarial Network for raw audio waveform generation. It is designed to learn directly from waveforms, producing realistic synthetic audio based on a learned distribution.

1. Overview
WaveGAN is a 1D convolutional GAN architecture that:
Learns to generate audio signals (e.g., speech, environmental sounds) without relying on spectrograms.


Uses transposed 1D convolutions in the Generator to upsample latent vectors into audio waveforms.


Uses standard 1D convolutions in the Discriminator to classify real vs. fake waveforms.


The model is trained adversarially using a dataset of real audio waveforms.

2. Dataset
This implementation uses the ESC-50 dataset, a labeled collection of environmental audio recordings (e.g., dog barking, rain, clock ticking).
Features:
50 audio classes


Each file is 5 seconds long, sampled at 44.1kHz


During training:
Audio is resampled to 16kHz


Converted to mono


Randomly cropped or padded to a fixed length (16,384 samples)


Automatic Download:
The dataset is automatically downloaded and extracted when running the script. No manual steps are required.
It will be saved to:
DATA/dcgan_dataset/training/data/esc50_dataset/audio/


3. Training the WaveGAN Model
To begin training, use the following command:
python ./codeP/main.py --dataset ESC-50 --model wavegan --otherOptions default

Or run the provided shell script:
bash codeB/wavegan.sh

This will:
Download and prepare the ESC-50 dataset


Initialize the generator and discriminator networks


Train the model for a number of epochs (default: 50)


Save generated audio samples and model checkpoints



4. Output and Sample Generation
The model provides the following functionalities:
a. Training Logs
During training, the following losses are logged for each batch:
G_loss – Generator loss (how well it fools the discriminator)


D_loss – Discriminator loss (how well it distinguishes real and fake audio)


These are displayed via a live progress bar (tqdm) and printed at the end of each epoch.
b. Generated Samples
At intervals (e.g., every 10 epochs), the model:
Generates sample audio waveforms


Saves them as .wav files using torchaudio


Stores them in:


generated_samples/sample_epoch_XX.wav

c. Final Output
After training, the model:
Generates a few final audio samples


Saves them in:


final_samples/generated_sample_1.wav
final_samples/generated_sample_2.wav
...

d. Model Checkpoints
Saved after training:
wavegan_generator.pth
wavegan_discriminator.pth

These can be reused for generating more samples without retraining.

5. Output Directory Structure
DATA/
└── dcgan_dataset/
    └── training/
        └── data/
            └── esc50_dataset/
                └── audio/                    # Raw WAV files from ESC-50

RESULTS/
├── generated_samples/                       # Epoch-based audio samples
└── final_samples/                           # Final audio samples
wavegan_generator.pth                        # Trained generator
wavegan_discriminator.pth                    # Trained discriminator


6. Default Hyperparameters
Parameter
Value
Latent Dimension
100
Audio Length
16,384
Batch Size
16
Learning Rate
0.0001
Optimizer
Adam
Epochs
50

These can be modified in the main() function in the WaveGAN script.

7. Notes
The model is compatible with  GPU.


If your machine does not support audio playback, you can visualize waveforms using libraries like matplotlib.


To generate custom samples, use the generate_samples() function after training is complete.






DCGAN Module
The Deep Convolutional Generative Adversarial Network (DCGAN) is a widely used GAN architecture designed to generate synthetic images from random noise using convolutional and transposed convolutional layers.

1. Overview
DCGAN consists of two adversarial networks:
Generator: Uses transposed convolutions to upsample a noise vector into a 64×64 color image.


Discriminator: Classifies input images as real or fake using downsampling convolutions.


These networks are trained in opposition to one another: the generator tries to create convincing fake images, while the discriminator learns to distinguish them from real data.

2. Dataset
This implementation uses the CIFAR-10 dataset, which contains 60,000 32×32 color images across 10 classes. Images are resized and cropped to 64×64 pixels to fit the DCGAN architecture.
The dataset is automatically downloaded to:
DATA/dcgan_dataset/training/data/

You do not need to manually download the dataset. It is handled by torchvision.datasets.

3. Training the DCGAN
To start training, run:
python ./codeP/main.py --dataset CIFAR10 --model dcgan --otherOptions default

Or execute the shell script:
bash codeB/dcgan.sh

This will:
Load CIFAR-10 images


Train the DCGAN for the specified number of epochs


Save training progress and generated sample images


Training logs include:
Generator and Discriminator loss per iteration


D(x) and D(G(z)) scores for monitoring progress



4. Output and Sample Generation
During training, the generator creates synthetic image samples using a fixed noise vector. These are saved periodically and can be visualized using torchvision.utils.make_grid.
After training, the model can generate new image samples using:
dcgan.generate_samples(num_samples=64)


5. Default Hyperparameters
Parameter
Value
Latent Dimension
100
Image Size
64×64
Batch Size
128
Learning Rate
0.0002
Optimizer
Adam
Epochs
5 (default)

These values are defined in the DCGAN class in codeP/models/dcgan.py and can be modified for experimentation.

6. Output Directory Structure
Generated images and models can be saved in a folder like:
RESULTS/
└── dcgan/
    ├── sample_epoch_01.png
    ├── sample_epoch_02.png
    └── ...

Model weights can be saved manually using torch.save() if desired.

GPT-Neo Module (File Loader Utility)
The GPT-Neo module includes a utility to load .txt training data from a directory for language modeling tasks.

1. Function: get_dataset(directory_path, file_type="")
This function is used to:
Load all .txt files from a specified directory


Print detailed logs (number of files, preview of text, errors, etc.)


Return a list of non-empty text samples


It ensures all training text is cleanly and completely loaded before being passed into a tokenizer or fine-tuning pipeline.

2. Usage Example
from codeP.models.gptneo import get_dataset

data = get_dataset('./DATA/gptneo_dataset/training/', file_type='training')

3. Logs Output
The function provides logs such as:
=== LOADING TRAINING FILES ===
Found 10 .txt files in ./DATA/gptneo_dataset/training/
✓ Loaded successfully - 853 characters
First 100 characters: The Industrial Revolution was a period of great change...
...
Total training texts loaded: 10

This makes it easier to debug data preparation errors in NLP workflows.

4. Notes
GPT-Neo is typically fine-tuned using Hugging Face's transformers library.


The provided utility supports the data preprocessing step; you may integrate it with tokenization and model fine-tuning functions.











