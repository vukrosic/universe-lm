import torch
import torch.optim as optim
from models.vqvae import VQVAE
from data.vqvae_dataset import get_vqvae_dataloader
import os

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    num_hiddens = 128
    num_residual_layers = 2
    num_residual_hiddens = 32
    num_embeddings = 1024 # Small codebook for efficiency
    embedding_dim = 64
    commitment_cost = 0.25
    learning_rate = 1e-3
    batch_size = 64
    num_epochs = 10
    
    dataloader = get_vqvae_dataloader(batch_size=batch_size)
    
    model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim, commitment_cost).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_recon_error = 0
        total_perplexity = 0
        
        for i, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            
            vq_loss, data_recon, perplexity = model(data)
            recon_error = torch.mean((data_recon - data)**2)
            loss = vq_loss + recon_error
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_error += recon_error.item()
            total_perplexity += perplexity.item()
            
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Recon: {recon_error.item():.4f}, Perplexity: {perplexity.item():.4f}")
        
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/vqvae_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()
