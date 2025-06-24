Variational Autoencoder (VAE) from Scratch in PyTorch
📁 VAE
├── config.py         # Configurasyon parametreleri
├── model.py          # VAE architecture
├── inference.py      # Inference kod
├── train.py          # Train kod

📦 Requirements
pip install torch torchvision numpy matplotlib

🧪 Training
python train.py config.py --checkpoint-path /path/to/vae.pth

🚀 Inference
python inference.py --checkpoint-path /path/to/vae.pth
