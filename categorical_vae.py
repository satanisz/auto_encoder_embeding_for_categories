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
import pickle

PRINT_EVERY = 10
SAVE_EVERY = 20

class EmployeeDataGenerator:
    """
    Generates synthetic employee data with logical relationships between features
    to provide meaningful patterns for the autoencoder to learn.
    """
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed=seed)

    def generate(self, n_samples: int = 2000) -> pd.DataFrame:
        departments = ['IT', 'HR', 'Marketing', 'Sales']
        levels = ['Junior', 'Mid', 'Senior']
        miasta = [
            'Warszawa', 
            'Krakow', 
            'Gdansk', 
            'Wroclaw', 
            'Poznan', 
            'Szczecin', 
            'Lodz', 
            'Bydgoszcz', 
            'Torun', 
            'Rzeszow', 
            'Lublin', 
            'Olsztyn', 
            'Bialystok', 
            'Katowice', 
            'Kielce', 
            'Opole', 
            'Zielona Gora', 
            'Gorzow Wielkopolski', 
            'Koszalin', 
            'Legnica', 
            'Radom', 
            'Tarnow', 
            'Elblag', 
            'Piotrkow Trybunalski', 
            'Kalisz', 
            'Konin', 
            'Jelenia Gora', 
            'Siedlce', 
            'Inowroclaw', 
            'Suwalki', 
            'Zamosc', 
            'Tarnobrzeg', 
            'Krosno', 
            'Slupsk', 
            'Nowy Targ', 
            'Ostróda', 
            'Wejherowo', 
            'Lask', 
            'Zgierz', 
            'Tomaszow Mazowiecki', 
            'Zgierz'
        ]
        
        dzial = self.rng.choice(departments, n_samples)
        poziom = self.rng.choice(levels, n_samples)
        miasto = self.rng.choice(miasta, n_samples)
        
        staz_lata = np.zeros(n_samples)
        is_junior = poziom == 'Junior'
        is_mid = poziom == 'Mid'
        is_senior = poziom == 'Senior'
        
        staz_lata[is_junior] = self.rng.uniform(0.5, 3.0, is_junior.sum())
        staz_lata[is_mid] = self.rng.uniform(2.0, 8.0, is_mid.sum())
        staz_lata[is_senior] = self.rng.uniform(5.0, 15.0, is_senior.sum())
        
        base_salary = np.zeros(n_samples)
        base_salary[is_junior] = 4000.0
        base_salary[is_mid] = 7000.0
        base_salary[is_senior] = 12000.0
        
        dept_mult = np.ones(n_samples)
        dept_mult[dzial == 'IT'] = 1.3
        dept_mult[dzial == 'Sales'] = 1.1
        dept_mult[dzial == 'HR'] = 0.9
        # Marketing is 1.0 (default)
        
        wynagrodzenie = base_salary * dept_mult + (staz_lata * 500.0) + self.rng.normal(0, 500.0, n_samples)
        
        return pd.DataFrame({
            'Dzial': dzial,
            'Poziom': poziom,
            'Staz_Lata': staz_lata,
            'Wynagrodzenie': wynagrodzenie,
            'City': miasto
        })

def prepare_data(df: pd.DataFrame) -> tuple[DataLoader, list[tuple[int, int]], int, dict[str, LabelEncoder], StandardScaler, torch.Tensor, torch.Tensor]:
    """Prepares the data by encoding categorical columns and scaling numerical columns."""
    cat_cols = ['Dzial', 'Poziom']
    encoders = {col: LabelEncoder() for col in cat_cols}
    for col in cat_cols:
        df[col] = encoders[col].fit_transform(df[col])

    num_cols = ['Staz_Lata', 'Wynagrodzenie']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    emb_dims = [(len(encoders[col].classes_), (len(encoders[col].classes_) // 2) + 1) for col in cat_cols]

    X_cat = torch.tensor(df[cat_cols].values, dtype=torch.long)
    X_cont = torch.tensor(df[num_cols].values, dtype=torch.float32)

    dataset = TensorDataset(X_cat, X_cont)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    return loader, emb_dims, len(num_cols), encoders, scaler, X_cat, X_cont


class TabularVAE(nn.Module):
    """
    A tabular Variational Autoencoder combining embedding layers for categorical features
    with linear layers for continuous data.
    """

    def __init__(self, emb_dims: list[tuple[int, int]], n_cont: int, latent_dim: int = 2) -> None:
        """
        Initializes the autoencoder components.
        """
        super().__init__()
        self.cat_dims = [ni for ni, nf in emb_dims]
        self.n_cont = n_cont
        self.embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_dims])
        n_emb = sum(nf for _, nf in emb_dims)
        
        self.layer_norm = nn.LayerNorm(n_emb + n_cont)
        
        self.encoder = nn.Sequential(
            nn.Linear(n_emb + n_cont, 708),
            nn.GELU(),
            nn.Linear(708, 177),
            nn.GELU(),
            nn.Linear(177, 59),
            nn.GELU(),
            nn.Linear(59, 7),
            nn.GELU()
        )
        self.fc_mu = nn.Linear(7, latent_dim)
        self.fc_var = nn.Linear(7, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 7),
            nn.GELU(),
            nn.Linear(7, 59),
            nn.GELU(),
            nn.Linear(59, 177),
            nn.GELU(),
            nn.Linear(177, 708),
            nn.GELU(),
            nn.Linear(708, n_cont + sum(self.cat_dims))
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning the reconstructed continuous tensor, 
        list of categorical logits, full input, mu, and logvar.
        """
        x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(x, dim=1)
        
        full_input = torch.cat([x, x_cont], dim=1)
        norm_input = self.layer_norm(full_input)
        
        hidden = self.encoder(norm_input)
        mu = self.fc_mu(hidden)
        logvar = self.fc_var(hidden)
        z = self.reparameterize(mu, logvar)
        
        decoded = self.decoder(z)
        
        reconstructed_cont = decoded[:, :self.n_cont]
        
        reconstructed_cat_logits = []
        start_idx = self.n_cont
        for dim in self.cat_dims:
            reconstructed_cat_logits.append(decoded[:, start_idx:start_idx+dim])
            start_idx += dim
            
        return reconstructed_cont, reconstructed_cat_logits, norm_input, mu, logvar


def train_model(loader: DataLoader, emb_dims: list[tuple[int, int]], n_cont: int, num_epochs: int = 1000) -> tuple[TabularVAE, list[float]]:
    """Trains the Tabular Variational Autoencoder model."""
    model = TabularVAE(emb_dims, n_cont=n_cont)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()

    print("Starting training...")
    losses = []

    for epoch in range(num_epochs):
        epoch_losses = []
        for cat_batch, cont_batch in loader:
            optimizer.zero_grad()
            recon_cont, recon_cat_logits, norm_input, mu, logvar = model(cat_batch, cont_batch)
            
            loss_cont = criterion_mse(recon_cont, cont_batch)
            
            loss_cat = 0
            for i in range(len(recon_cat_logits)):
                loss_cat += criterion_ce(recon_cat_logits[i], cat_batch[:, i])
                
            kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            
            loss = loss_cont + loss_cat + kld_loss
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        
        if (epoch + 1) % PRINT_EVERY == 0:
            print(f"Epoch {epoch+1:02} | Loss: {avg_loss:.4f}")


    print("\nTraining complete. The model has learned the data representations.")
    return model, losses


def evaluate_model(model: TabularVAE, X_cat: torch.Tensor, X_cont: torch.Tensor) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Evaluates the model to extract the latent space and reconstructions."""
    model.eval()
    with torch.no_grad():
        x_emb = [emb(X_cat[:, i]) for i, emb in enumerate(model.embeddings)]
        x_emb = torch.cat(x_emb, dim=1)
        full_input = torch.cat([x_emb, X_cont], dim=1)
        norm_input = model.layer_norm(full_input)
        
        hidden = model.encoder(norm_input)
        mu = model.fc_mu(hidden)
        latent_space = mu.numpy()
        
        recon_cont, recon_cat_logits, _, _, _ = model(X_cat, X_cont)
        recon_cat_preds = [torch.argmax(logits, dim=1).numpy() for logits in recon_cat_logits]
    
    return latent_space, recon_cont.numpy(), recon_cat_preds


def plot_results(losses: list[float], df: pd.DataFrame, latent_space: np.ndarray, 
                 recon_cont: np.ndarray, encoders: dict[str, LabelEncoder], emb_dims: list[tuple[int, int]]) -> None:
    """Generates and saves all visualizations."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
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

    original_salary = df['Wynagrodzenie']
    # recon_cont[:, 1] is Wynagrodzenie since num_cols = ['Staz_Lata', 'Wynagrodzenie']
    reconstructed_salary = recon_cont[:, 1]

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

    original_staz = df['Staz_Lata']
    reconstructed_staz = recon_cont[:, 0]

    plt.figure(figsize=(8, 8))
    plt.scatter(original_staz, reconstructed_staz, alpha=0.5)
    plt.plot([original_staz.min(), original_staz.max()], 
             [original_staz.min(), original_staz.max()], 'r--')
    plt.title('Original vs Reconstructed staz (Scaled)')
    plt.xlabel('Original staz')
    plt.ylabel('Reconstructed staz')
    plt.grid(True)
    plt.savefig('reconstruction_staz.png')
    print("Saved reconstruction_staz.png")
    plt.close()


def run_pipeline() -> None:
    """Main pipeline execution function."""
    generator = EmployeeDataGenerator(seed=42)
    df = generator.generate(n_samples=2000)
    
    loader, emb_dims, n_cont, encoders, scaler, X_cat, X_cont = prepare_data(df)
    
    model, losses = train_model(loader, emb_dims, n_cont, num_epochs=1000)
    
    latent_space, recon_cont, recon_cat_preds = evaluate_model(model, X_cat, X_cont)
    
    plot_results(losses, df, latent_space, recon_cont, encoders, emb_dims)
    
    with open("final_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Final model saved to final_model.pkl")


def run_model() -> pd.DataFrame:
    """Runs the model to evaluate anomalies and returns a dataframe with the top 10 anomalies."""
    with open("final_model.pkl", "rb") as f:
        model: TabularVAE = pickle.load(f)
        
    generator = EmployeeDataGenerator(seed=42)
    df = generator.generate(n_samples=2000)
    
    df_prepared = df.copy()
    _, _, _, _, _, X_cat, X_cont = prepare_data(df_prepared)
    
    model.eval()
    criterion_ce = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        recon_cont, recon_cat_logits, _, mu, logvar = model(X_cat, X_cont)
        
        loss_cont = torch.mean((recon_cont - X_cont)**2, dim=1)
        
        loss_cat = torch.zeros(X_cat.size(0))
        for i in range(len(recon_cat_logits)):
            loss_cat += criterion_ce(recon_cat_logits[i], X_cat[:, i])
            
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        anomaly_scores = loss_cont + loss_cat + kld_loss
        
    df['anomaly_score'] = anomaly_scores.numpy()
    df['Latent_X'] = mu[:, 0].numpy()
    df['Latent_Y'] = mu[:, 1].numpy()
    
    top_10_anomalies = df.nlargest(10, 'anomaly_score')
    
    print("Top 10 Anomalies:")
    print(top_10_anomalies)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Latent_X', y='Latent_Y', color='lightgray', alpha=0.5, label='Normal')
    sns.scatterplot(data=top_10_anomalies, x='Latent_X', y='Latent_Y', color='red', s=150, edgecolor='black', marker='X', label='Anomaly')
    
    for i, row in top_10_anomalies.iterrows():
        plt.text(row['Latent_X'] + 0.05, row['Latent_Y'] + 0.05, str(i), fontsize=9, color='darkred')
        
    plt.title('Top 10 Anomalies in Latent Space')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('anomalies_latent_space.png')
    print("Saved anomalies_latent_space.png")
    
    return top_10_anomalies

if __name__ == "__main__":
    run = False
    if run:
        run_pipeline()
    else:
        run_model()