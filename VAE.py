import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# Define the encoder network
# ------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# ------------------------------
# Define the decoder network
# ------------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        x_hat = self.sigmoid(self.fc2(h))
        return x_hat

# ------------------------------
# VAE model combining encoder and decoder
# ------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

# ------------------------------
# Loss: Reconstruction + KL divergence
# ------------------------------
def loss_function(x_hat, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# ------------------------------
# Data loading (MNIST)
# ------------------------------
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# ------------------------------
# Model setup
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------
# Training Loop
# ------------------------------
epochs = 10
model.train()
for epoch in range(epochs):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, 784).to(device)
        optimizer.zero_grad()
        x_hat, mu, logvar = model(data)
        loss = loss_function(x_hat, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f"Epoch {epoch+1}, Average Loss: {train_loss / len(train_loader.dataset):.4f}")

# ------------------------------
# 1. Manual walkthrough on one sample
# ------------------------------
model.eval()
with torch.no_grad():
    sample, _ = next(iter(train_loader))
    sample = sample[0].view(-1, 784).to(device)  # one digit

    # Step-by-step forward pass
    mu, logvar = model.encoder(sample)
    z = model.reparameterize(mu, logvar)
    reconstruction = model.decoder(z)
    
    print("Manual Forward Pass:")
    print(f"Mean vector (mu): {mu.squeeze().cpu().numpy()}")
    print(f"Log-variance vector (logvar): {logvar.squeeze().cpu().numpy()}")
    print(f"Sampled latent vector (z): {z.squeeze().cpu().numpy()}")

    # Show input and reconstruction
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(sample.view(28, 28).cpu(), cmap='gray')
    axs[0].set_title("Original")
    axs[1].imshow(reconstruction.view(28, 28).cpu(), cmap='gray')
    axs[1].set_title("Reconstructed")
    for ax in axs: ax.axis("off")
    plt.suptitle("Manual Forward Pass Visualization")
    plt.show()

# ------------------------------
# 2. Effect of changing latent variables
# ------------------------------
print("Visualizing the effect of changing one latent dimension...")

with torch.no_grad():
    # Fix the latent vector from a real digit
    mu, logvar = model.encoder(sample)
    z = model.reparameterize(mu, logvar)

    # Vary one latent variable (e.g., z[0]) across a range
    z = z.squeeze().cpu().numpy()
    fig, axes = plt.subplots(1, 7, figsize=(14, 2))

    for i, val in enumerate(np.linspace(-3, 3, 7)):
        z_mod = z.copy()
        z_mod[0] = val  # modify the first dimension
        z_tensor = torch.tensor(z_mod).unsqueeze(0).to(device)
        img = model.decoder(z_tensor).view(28, 28).cpu()
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"z[0]={val:.1f}")
        axes[i].axis("off")
    plt.suptitle("Effect of Changing Latent Dimension z[0]")
    plt.tight_layout()
    plt.show()

# ------------------------------
# 3. Random latent space sampling
# ------------------------------
with torch.no_grad():
    z = torch.randn(16, 20).to(device)
    samples = model.decoder(z).view(-1, 1, 28, 28).cpu()

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].squeeze(), cmap="gray")
        ax.axis("off")
    plt.suptitle("Generated Samples from Latent Space")
    plt.tight_layout()
    plt.show()

