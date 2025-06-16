import time
import math
import torch
import torch.nn as nn
import numpy as np
from Model import Model
from Dataset import get_dataloaders

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Hyperparameters
'''
batch_size = 128
learning_rate = 0.001
min_lr = 0.0001
epochs = 50
weight_decay = 0.00001
window_size = 50
k_size = 5
dim = 32
r = 2
num_blocks = 2
max_rul = 125
train_workers = 4
'''
Data loader
'''
# 'FD001', 'FD002', 'FD003', 'FD004'
name = 'FD004'
train_file = f'cmapss/train_{name}.txt'
test_file = f'cmapss/test_{name}.txt'
truth_file = f'cmapss/RUL_{name}.txt'

train_set, test_set = get_dataloaders(
    train_file=train_file,
    test_file=test_file,
    truth_file=truth_file,
    window_size=window_size,
    train_batch=batch_size,
    train_workers=train_workers
)
_, test_cpu = get_dataloaders(
    train_file=train_file,
    test_file=test_file,
    truth_file=truth_file,
    window_size=window_size,
    train_batch=1,
    train_workers=0
)
'''
Training cycle
'''
NN = Model(k_size, dim, r, window_size, num_blocks).to(device)
optimizer = torch.optim.AdamW(NN.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss = nn.MSELoss()
_lambda = lambda epoch: max(min_lr, min_lr + 0.5 * (1 + math.cos(epoch / epochs * math.pi)))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lambda)


def cmapss_score(pred, true):
    diff = pred - true
    score = np.where(
        diff < 0,
        np.exp(-diff / 10) - 1,
        np.exp(diff / 13) - 1
    )
    return np.sum(score)


def train_epoch(model, loader, opt_fn, loss):
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        opt_fn.zero_grad()
        preds = model(inputs).squeeze()
        loss_fn = loss(preds, targets)
        loss_fn.backward()
        opt_fn.step()
        total_loss += loss_fn.item() * inputs.size(0)
    return total_loss / len(loader.dataset)


def eval_model(model, test_loader, device):
    np.random.seed(42)
    torch.set_num_threads(1)
    model.to(device)
    model.eval()

    predictions = []
    true_values = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs).cpu().numpy()
            predictions.extend(preds)
            true_values.extend(targets.numpy())

    predictions = np.array(predictions)
    true_values = np.array(true_values)

    rmse = np.sqrt(np.mean((predictions - true_values) ** 2))
    score = cmapss_score(predictions, true_values)
    return score, rmse


def main():
    for epoch in range(epochs):
        train = train_epoch(NN, train_set, optimizer, loss)
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print(f'Epoch: {epoch + 1}/{epochs}, MSE loss: {train:.4f}, learning rate: {before_lr:.6f} -> {after_lr:.6f}')

    start_time = time.time()
    cpu = torch.device('cpu')
    test_score, test_rmse = eval_model(NN, test_cpu, cpu)
    elapsed = time.time() - start_time
    print(f"Evaluation time = {(elapsed * 1000):.2f} ms")
    print(f"NASA Score: {test_score:.2f}, RMSE: {test_rmse:.2f}")

    torch.save(NN, f'RULformer_{name}.pt')
    print(f"Model saved to RULformer_{name}.pt")


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()
