import sys
import importlib

import torch
from torchvision.utils import save_image


def get_everything(config_path, args):
    module_path = config_path.replace('/', '.').replace('.py', '')
    module = importlib.import_module(module_path, package=None)
    return module.everything(args)


def show_reconstructions(model, data_loader, device, num_images=8):
    model.eval()
    x, _ = next(iter(data_loader))
    x = x.to(device)[:num_images]
    with torch.no_grad():
        x_recon, _, _ = model(x)

    x = x.cpu()
    x_recon = x_recon.cpu()

    return x_recon


def generate_samples(model, device, latent_dim, num_samples=8):
    model.eval()
    z = torch.randn(num_samples, latent_dim).to(device)
    with torch.no_grad():
        x_gen = model.decoder(model.fc_decode(z).view(-1, 512, 3, 3)).cpu()

    x_gen = (x_gen + 1) / 2

    return x_gen


def main(config_path, args):
    evy = get_everything(config_path, args)

    latent_dim = evy['latent_dim']
    device = evy['device']

    checkpoint_path = evy['checkpoint_path']

    model = torch.load(checkpoint_path, map_location=device, weights_only=False)

    x_gen = generate_samples(model, device, latent_dim)

    for i in range(x_gen.shape[0]):
        save_image(x_gen[i], f'Generated_image{i}.png')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1], sys.argv[2:])

