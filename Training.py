import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from Model import Model
from Layers import Conv1DMHSA, MHSA
from Dataset import get_dataloaders

seed = ...
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Hyperparameters
'''
batch_size = 128
learning_rate = 0.001
min_lr = 0.00001
epochs = 120
weight_decay = 0.00001
window_size = 60
k_size = 5
dim = 64
r = 8
num_blocks = 2
max_rul = 125
train_workers = 0
func_type = 'sketch'  # 'sketch' 'softmax'
'''
Data loader
'''
name = 'FD002'
train_file = f'cmapss/train_{name}.txt'
test_file = f'cmapss/test_{name}.txt'
truth_file = f'cmapss/RUL_{name}.txt'

train_set, _, val_set = get_dataloaders(train_file=train_file, test_file=test_file, truth_file=truth_file,
                                        window_size=window_size, train_batch=batch_size, train_workers=train_workers)

'''
Training cycle
'''
NN = Model(k_size, dim, r, window_size, num_blocks, func_type).to(device)
optimizer = torch.optim.AdamW(NN.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

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


def eval_model(model, loader):
    torch.set_num_threads(1)
    model.eval()

    predictions = []
    true_values = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs).squeeze()
            predictions.append(outputs.cpu())
            true_values.append(targets.cpu())

        preds = torch.cat(predictions)
        trues = torch.cat(true_values)

    mse = torch.mean((preds - trues) ** 2)
    rmse = torch.sqrt(mse).item()
    return mse, rmse


def main():
    train_losses, val_losses, val_rmses = [], [], []

    for epoch in range(epochs):
        train_loss = train_epoch(NN, train_set, optimizer, loss)
        train_losses.append(train_loss)
        parameter_logger.log_epoch(NN, epoch)

        val_mse, epoch_rmse = eval_model(NN, val_set)
        val_losses.append(val_mse)
        val_rmses.append(epoch_rmse)

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print(
            f'Epoch: {epoch + 1}/{epochs}, MSE: {train_loss:.4f}, Val.MSE: {val_mse:.4f}, Val.RMSE: {epoch_rmse:.4f}, '
            f'learning rate: {before_lr:.6f} -> {after_lr:.6f}')

    actual_epochs = len(train_losses)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, actual_epochs + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, actual_epochs + 1), val_losses, 'r-', label='Validation Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_{name}.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, actual_epochs + 1), train_losses, 'b-', label='Validation RMSE')
    plt.title('RMSE per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'validation_RMSE_{name}.png')
    plt.show()

    parameter_logger.plot()

    torch.save(NN, f'RULformer_{name}_{func_type}.pt')
    print(f"Model saved to RULformer_{name}_{func_type}.pt")


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()
