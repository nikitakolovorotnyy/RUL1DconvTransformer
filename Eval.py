import time
import torch
import torch.profiler
import numpy as np
from Dataset import get_dataloaders
import matplotlib.pyplot as plt

name = 'FD002'
func_type = 'sketch'  # 'sketch' 'softmax'
train_file = f'cmapss/train_{name}.txt'
test_file = f'cmapss/test_{name}.txt'
truth_file = f'cmapss/RUL_{name}.txt'
model_path = f'RULformer_{name}_{func_type}.pt'

_, test_cpu, _ = get_dataloaders(train_file=train_file, test_file=test_file, truth_file=truth_file, window_size=60,
                                 train_batch=1, train_workers=0)


def plot_predictions(predictions: np.ndarray, true_values: np.ndarray):
    plt.figure(figsize=(12, 6))
    plt.plot(true_values,   label='True RUL',      linewidth=2)
    plt.plot(predictions,   label='Predicted RUL', linewidth=2, linestyle='--')
    plt.xlabel('Engine ID')
    plt.ylabel('RUL')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'RUL_predictions_{name}.jpg')
    plt.show()

def cmapss_score(pred, true):
    diff = pred - true
    score = np.where(
        diff < 0,
        np.exp(-diff / 10) - 1,
        np.exp(diff / 13) - 1
    )
    return np.sum(score)

def eval_model(model, test_loader, device):
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
    return score, rmse, predictions, true_values

def main():
    device = torch.device('cpu')
    model: torch.nn.Module = torch.load(model_path, map_location=device)
    model.to(device)

    sample_input, _ = next(iter(test_cpu))
    sample_input = sample_input.to(device)

    # 3) Трассировка + заморозка
    with torch.no_grad():
        traced = torch.jit.trace(model, sample_input)
    traced = torch.jit.freeze(traced)

    print("\n=== Profiling one forward pass ===")
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True
        ) as prof:
        with torch.no_grad():
            _ = traced(sample_input)
    print(prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=10
    ))

    start_time = time.time()
    cpu = torch.device('cpu')
    test_score, test_rmse, pred, true = eval_model(traced, test_cpu, cpu)
    elapsed = time.time() - start_time
    print(f"Evaluation time = {(elapsed * 1000):.2f} ms")
    print(f"NASA Score: {test_score:.2f}, RMSE: {test_rmse:.2f}")

    plot_predictions(pred, true)


if __name__ == '__main__':
    main()
