import sys
import importlib

import torch


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kl_div


def get_everything(config_path, args):
    module_path = config_path.replace('/', '.').replace('.py', '')
    module = importlib.import_module(module_path, package=None)
    return module.everything(args)


def main(config_path, args):
    evy = get_everything(config_path, args)

    train_loader = evy['train_loader']

    model = evy['model']
    optimizer = evy['optimizer']
    epochs = evy['epochs']
    device = evy['device']

    run = evy['run']

    checkpoint_path = evy['checkpoint_path']

    model.train()
    for epoch in range(epochs):
        for x, _ in train_loader:
            x = x.to(device)
            recon_x, mu, logvar = model(x)
            recon_loss, kl_div = vae_loss(recon_x, x, mu, logvar)
            loss = recon_loss + kl_div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            run.log(
                {
                    "reconstruct_loss": recon_loss.item(),
                    "kl_loss": kl_div.item(),
                    "total_loss": loss,
                    "epoch": epoch
                }
            )

        torch.save(model, checkpoint_path)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1], sys.argv[2:])
