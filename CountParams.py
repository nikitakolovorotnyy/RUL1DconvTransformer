import torch
from Model import Model
from ptflops import get_model_complexity_info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model(5, 32, 2, 50, 2).to(device)
input_tsr = (50, 24)
flops, params = get_model_complexity_info(model, input_tsr, as_strings=True, print_per_layer_stat=True, verbose=True)
print(f"FLOPs: {flops}")
print(f"Params: {params}")


def count_layers(model):
    return len(list(model.modules()))

print(f'Number of layers: {count_layers(model)}')

total_params = sum(p.numel() for p in model.parameters())

print(f"All params: {total_params:,}")
