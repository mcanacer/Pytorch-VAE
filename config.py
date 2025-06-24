import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb
from model import VAE


def parse_args(args):
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--image-size', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=2)

    # Model
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--num-epochs', type=int, default=100)

    # Wandb
    parser.add_argument('--project', type=str, default='VAE')
    parser.add_argument('--name', type=str, default='run_standard')

    # Save
    parser.add_argument('--checkpoint-path', type=str, required=True)

    return parser.parse_args(args)


def everything(args):
    args = parse_args(args)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),  # Normalize [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale [-1, 1]
    ])

    train_dataset = datasets.CelebA(
        root='./data',
        split='train',
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    epochs = args.num_epochs

    run = wandb.init(
        project=args.project,
        name=args.name,
        reinit=True,
        config=vars(args)
    )

    return {
        'train_loader': train_loader,
        'model': model,
        'optimizer': optimizer,
        'latent_dim': args.latent_dim,
        'epochs': epochs,
        'device': device,
        'run': run,
        'checkpoint_path': args.checkpoint_path,
    }

