import torch
import numpy as np
from src.datasets import TrajectoryDataset, read_split
from src.models_lstm import read_config, model_from_config

# paths
split_path = "data/folds.json"
config_path = "model_configs/lstm_128_3_30.json"
model_path = "experiments/models/lstm_run/model_20260416_051629.pt"  # latest model

# load config + model
config = read_config(config_path)
model = model_from_config(config, 3)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# load data
split = read_split(split_path)
fold = split.folds[0]

X_len = 20
y_len = config.prediction_sequence_length

test_dataset = TrajectoryDataset(split.test, X_len, y_len)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10)

# evaluation
mse_list = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        mse = torch.mean((outputs - labels) ** 2).item()
        mse_list.append(mse)

mse = np.mean(mse_list)
rmse = np.sqrt(mse)

print("MSE:", mse)
print("RMSE:", rmse)

