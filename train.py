
import torch
from configs.dit_config import DiTConfig
from models.dit import DiT
from data.video_loader import get_dataloaders
from training.video_trainer import train_video_model

def main():
    config = DiTConfig()
    # Faster training for testing
    config.train_steps = 100
    config.log_every = 10
    config.save_every = 50 
    config.batch_size = 1 # Reduce batch size to 1 to debug OOM
    config.hidden_size = 384 # Reduce model size (DiT-S/2 equivalent-ish)
    config.depth = 12
    config.num_heads = 6

    
    print("Initializing Model...")
    model = DiT(config)
    
    print("Getting Dataloaders...")
    train_loader, val_loader = get_dataloaders(config)
    
    print("Starting Training...")
    train_video_model(model, train_loader, val_loader, config)
    
if __name__ == "__main__":
    main()
