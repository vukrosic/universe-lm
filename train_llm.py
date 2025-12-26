import argparse
import torch
from configs.dit_config import DiTConfig
from models.dit import DiT
from data.video_loader import get_dataloaders
from training.video_trainer import train_video_model
from utils.logger import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Train DiT Video Model")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--steps", type=int, help="Training steps")
    args = parser.parse_args()
    
    config = DiTConfig()
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.steps:
        config.train_steps = args.steps
        
    logger = setup_logging()
    logger.info("Initializing Video DiT...")
    
    model = DiT(config)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_loader, val_loader = get_dataloaders(config)
    
    train_video_model(model, train_loader, val_loader, config)

if __name__ == "__main__":
    main()
