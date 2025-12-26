import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import os
# import wandb

def train_video_model(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    
    print(f"Starting training for {config.train_steps} steps...")
    
    step = 0
    model.train()
    
    pbar = tqdm(total=config.train_steps)
    
    while step < config.train_steps:
        for x, y in train_loader:
            if step >= config.train_steps:
                break
                
            x = x.to(device)
            y = y.to(device)
            
            # Flow Matching Training
            # 1. Sample t
            t = torch.rand(x.shape[0], device=device)
            
            # 2. Sample x_0 (noise)
            x_0 = torch.randn_like(x)
            x_1 = x # Data
            
            # 3. Interpolate
            # x_t = (1 - t) * x_0 + t * x_1
            # We need to broadcast t: [B] -> [B, 1, 1, 1, 1]
            t_reshaped = t.view(-1, 1, 1, 1, 1)
            x_t = (1 - t_reshaped) * x_0 + t_reshaped * x_1
            
            # 4. Target velocity
            v_target = x_1 - x_0
            
            # 5. Predict
            with torch.cuda.amp.autocast():
                v_pred = model(x_t, t, y)
                if v_pred.shape[1] != v_target.shape[1] and v_pred.shape[1] == v_target.shape[1] * 2:
                    v_pred, _ = torch.split(v_pred, v_target.shape[1], dim=1)
                loss = F.mse_loss(v_pred, v_target)
            
            # 6. Optimize
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loss_val = loss.item()
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
            pbar.update(1)
            
            if step % config.log_every == 0:
                # Log to wandb or print
                pass
                
            if step % config.save_every == 0 and step > 0:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), f"checkpoints/dit_{step}.pt")
                
            step += 1
