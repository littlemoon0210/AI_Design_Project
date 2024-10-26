import os
import torch
import numpy as np

def set_seed(seed=42):
    """
    Set the random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, optimizer, path, iter_num, best_val_loss, model_args, config):
    """
    Save model checkpoint.
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'model_args': model_args,
        'config': config,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path, device='cpu'):
    """
    Load model checkpoint.
    """
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        model_args = checkpoint['model_args']
        config = checkpoint['config']
        print(f"Loaded checkpoint from {path} (iter {iter_num})")
        return model, optimizer, iter_num, best_val_loss, model_args, config
    else:
        print(f"No checkpoint found at {path}")
        return model, optimizer, 0, float('inf'), None, None

def estimate_loss(model, data_loader, device, eval_iters=100):
    """
    Estimate the model loss over a given dataset.
    """
    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = next(data_loader)
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            losses.append(loss.item())

    model.train()
    return np.mean(losses)

def create_data_loader(data, batch_size, block_size, device='cpu'):
    """
    Create a simple data loader for training and evaluation.
    """
    def get_batch():
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    return get_batch

def configure_optimizer(model, learning_rate, weight_decay, betas, device_type='cpu'):
    """
    Configure the AdamW optimizer.
    """
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    use_fused = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames and device_type == 'cuda'
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
    return optimizer

def print_training_info(iter_num, loss, val_loss=None, lr=None):
    """
    Print the training info including iteration, loss, and learning rate.
    """
    if val_loss is not None and lr is not None:
        print(f"Iter {iter_num}: Loss {loss:.4f}, Val Loss {val_loss:.4f}, LR {lr:.6f}")
    elif val_loss is not None:
        print(f"Iter {iter_num}: Loss {loss:.4f}, Val Loss {val_loss:.4f}")
    elif lr is not None:
        print(f"Iter {iter_num}: Loss {loss:.4f}, LR {lr:.6f}")
    else:
        print(f"Iter {iter_num}: Loss {loss:.4f}")

def adjust_learning_rate(optimizer, lr):
    """
    Adjust the learning rate of the optimizer.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
