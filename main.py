"""
This module prepares tabular data, creates an Autoencoder (TabularAE) model with 
embedding layers for categorical features and linear layers for numerical features,
and then trains it to learn data representations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


class DataGenerator:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed=seed)

    def generate(self) -> pd.DataFrame:
        data = {
            'Dzial': ['IT', 'HR', 'IT', 'Marketing', 'Sales', 'IT', 'HR', 'Sales'] * 20,
            'Poziom': ['Junior', 'Senior', 'Mid', 'Junior', 'Senior', 'Mid', 'Senior', 'Junior'] * 20,
            'Staz_Lata': self.rng.uniform(1, 15, 160),
            'Wynagrodzenie': self.rng.uniform(4000, 15000, 160)
        }
        return pd.DataFrame(data)

generator = DataGenerator(seed=42)
df = generator.generate()

cat_cols = ['Dzial', 'Poziom']
encoders = {col: LabelEncoder() for col in cat_cols}
for col in cat_cols:
    df[col] = encoders[col].fit_transform(df[col])

num_cols = ['Staz_Lata', 'Wynagrodzenie']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

emb_dims = [(len(encoders[col].classes_), (len(encoders[col].classes_) // 2) + 1) for col in cat_cols]


class TabularAE(nn.Module):
    """
    A tabular Autoencoder combining embedding layers for categorical features
    with linear layers for continuous data.
    """

    def __init__(self, emb_dims: list[tuple[int, int]], n_cont: int, latent_dim: int = 2) -> None:
        """
        Initializes the autoencoder components.
        """
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_dims])
        n_emb = sum(nf for _, nf in emb_dims)
        
        self.encoder = nn.Sequential(
            nn.Linear(n_emb + n_cont, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, n_emb + n_cont)
        )

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning the reconstructed tensor and the full 
        input tensor representation.
        """
        x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(x, dim=1)
        
        full_input = torch.cat([x, x_cont], dim=1)
        
        latent = self.encoder(full_input)
        reconstructed = self.decoder(latent)
        return reconstructed, full_input


X_cat = torch.tensor(df[cat_cols].values, dtype=torch.long)
X_cont = torch.tensor(df[num_cols].values, dtype=torch.float32)

dataset = TensorDataset(X_cat, X_cont)
loader = DataLoader(dataset, batch_size=1024, shuffle=True)

model = TabularAE(emb_dims, n_cont=len(num_cols))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Starting training...")
losses = []

num_epochs = 1000

for epoch in range(num_epochs):
    epoch_losses = []
    for cat_batch, cont_batch in loader:
        optimizer.zero_grad()
        reconstructed, original = model(cat_batch, cont_batch)
        
        loss = criterion(reconstructed, original)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    losses.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:02} | Loss: {avg_loss:.4f}")

print("\nTraining complete. The model has learned the data representations.")

model.eval()
with torch.no_grad():
    x_emb = [emb(X_cat[:, i]) for i, emb in enumerate(model.embeddings)]
    x_emb = torch.cat(x_emb, dim=1)
    full_input = torch.cat([x_emb, X_cont], dim=1)
    
    latent_space = model.encoder(full_input).numpy()
    reconstructed, _ = model(X_cat, X_cont)

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), losses, marker='o', linestyle='-', color='b')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)
plt.savefig('training_loss.png')
print("Saved training_loss.png")
plt.close()

df['Latent_X'] = latent_space[:, 0]
df['Latent_Y'] = latent_space[:, 1]
df['Dzial_Name'] = encoders['Dzial'].inverse_transform(df['Dzial'])
df['Poziom_Name'] = encoders['Poziom'].inverse_transform(df['Poziom'])

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='Latent_X', y='Latent_Y', hue='Dzial_Name', style='Poziom_Name', s=100)
plt.title('2D Latent Space Representation')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('latent_space.png')
print("Saved latent_space.png")
plt.close()

n_emb = sum(nf for _, nf in emb_dims)
original_salary = df['Wynagrodzenie']
reconstructed_salary = reconstructed[:, n_emb + 1].numpy()

plt.figure(figsize=(8, 8))
plt.scatter(original_salary, reconstructed_salary, alpha=0.5)
plt.plot([original_salary.min(), original_salary.max()], 
         [original_salary.min(), original_salary.max()], 'r--')
plt.title('Original vs Reconstructed Salary (Scaled)')
plt.xlabel('Original Salary')
plt.ylabel('Reconstructed Salary')
plt.grid(True)
plt.savefig('reconstruction_salary.png')
print("Saved reconstruction_salary.png")
plt.close()

n_emb = sum(nf for _, nf in emb_dims)
original_salary = df['Staz_Lata']
reconstructed_salary = reconstructed[:, n_emb + 1].numpy()

plt.figure(figsize=(8, 8))
plt.scatter(original_salary, reconstructed_salary, alpha=0.5)
plt.plot([original_salary.min(), original_salary.max()], 
         [original_salary.min(), original_salary.max()], 'r--')
plt.title('Original vs Reconstructed staz (Scaled)')
plt.xlabel('Original staz')
plt.ylabel('Reconstructed staz')
plt.grid(True)
plt.savefig('reconstruction_staz.png')
print("Saved reconstruction_staz.png")
plt.close()