def get_model(name):
    if name == "dcgan":
        from .dcgan import DCGAN
        return DCGAN()
    elif name == "cgan":
        from .cgan import ConditionalGAN
        return ConditionalGAN(noise_dim=100,num_classes=10,img_channels=1,img_size=28)
    elif name == "cvae":
        from .cvae import ConditionalVAE
        return ConditionalVAE(num_classes=10,img_channels=1,img_size=28,latent_dim=20,beta=1.0)
    else:
        raise ValueError(f"Unknown model: {name}")
