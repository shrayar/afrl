import argparse
from datetime import datetime
import os

import torch
import torch.utils.tensorboard

from src.datasets import Fold, Split, TrajectoryDataset, read_split
from src.models_lstm import Config, model_from_config, read_config


class EarlyStopping:
    """
    Stops training when monitored metric hasn't improved for `patience` epochs.
    Optionally restores best model weights.
    """

    def __init__(self, patience=10, min_delta=0.0, mode="min", restore_best=True):
        assert mode in ("min", "max")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.best_score = None
        self.best_state = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def _is_better(self, score, best):
        if best is None:
            return True
        if self.mode == "min":
            return score < best - self.min_delta
        return score > best + self.min_delta

    def step(self, score, model=None):
        """
        Call this at the end of each epoch with the validation metric.
        Returns True if training should stop.
        """
        if self._is_better(score, self.best_score):
            self.best_score = score
            self.num_bad_epochs = 0
            if model is not None and self.restore_best:
                import copy
                self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore(self, model):
        """Restore best weights if available."""
        if self.restore_best and self.best_state is not None and model is not None:
            model.load_state_dict(self.best_state)


class Trainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: torch.nn.Module
    train_loader: torch.utils.data.DataLoader
    validation_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    device: torch.device
    writer: torch.utils.tensorboard.SummaryWriter

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        writer: torch.utils.tensorboard.SummaryWriter,
        model_path: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.device = device
        self.writer = writer
        self.model_path = model_path

        print("using device", self.device)
        os.makedirs(model_path, exist_ok=True)

    def train_epochs(self, epochs: int):
        best_vloss = float("inf")
        early_stopper = EarlyStopping(patience=10, min_delta=1e-5)

        for epoch in range(epochs):
            self.model.train(True)
            avg_loss = self.train_epoch(epoch)

            self.model.eval()
            running_vloss = 0.0

            with torch.no_grad():
                for i, vdata in enumerate(self.validation_loader):
                    vinputs, vlabels = vdata
                    vinputs = vinputs.to(self.device)
                    vlabels = vlabels.to(self.device)

                    voutputs = self.model(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss.item()

            avg_vloss = running_vloss / (i + 1)
            print(f"LOSS train {avg_loss} valid {avg_vloss}")

            self.writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": avg_loss, "Validation": avg_vloss},
                epoch + 1,
            )
            self.writer.flush()

            if early_stopper.step(avg_vloss, self.model):
                print("Early stopping")
                break

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_file = os.path.join(
                    self.model_path,
                    f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt",
                )
                torch.save(self.model.state_dict(), model_file)

        early_stopper.restore(self.model)

    def train_epoch(self, epoch_index: int):
        running_loss = 0.0
        last_loss = 0.0

        for i, data in enumerate(self.train_loader):
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if i % 1000 == 999:
                last_loss = running_loss / 1000
                print(f"batch {i + 1} loss: {last_loss}")
                tb_x = epoch_index * len(self.train_loader) + i + 1
                self.writer.add_scalar("Training Loss", last_loss, tb_x)
                running_loss = 0.0

        if len(self.train_loader) > 0 and running_loss > 0:
            remaining = len(self.train_loader) % 1000
            if remaining == 0:
                remaining = 1000
            last_loss = running_loss / remaining

        return last_loss


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM models with specific data and hyperparameters."
    )
    parser.add_argument(
        "name",
        type=str,
        help="The name of this training job. Used for tensorboard reporting.",
    )
    parser.add_argument(
        "-s",
        "--split",
        type=str,
        required=True,
        help="Path to a valid .json file that specifies the data split for k-fold CV.",
    )
    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        required=True,
        help="Specifies the specific fold to train on. Zero-indexed.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Specifies an LSTM model config file.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Optional path to a saved model checkpoint.",
    )

    args = parser.parse_args()

    split: Split = read_split(args.split)
    if args.fold >= len(split.folds):
        raise Exception("Invalid fold index.")

    fold: Fold = split.folds[args.fold]
    config: Config = read_config(args.config)

    X_len = 20
    y_len = config.prediction_sequence_length

    train_dataset = TrajectoryDataset(fold.train, X_len, y_len)
    validation_dataset = TrajectoryDataset(fold.validation, X_len, y_len)
    test_dataset = TrajectoryDataset(split.test, X_len, y_len)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_from_config(config, 3)

    if args.model:
        model.load_state_dict(torch.load(args.model, map_location=device))

    model.to(device)

    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        loss_fn=torch.nn.MSELoss(),
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        device=device,
        writer=torch.utils.tensorboard.SummaryWriter(f"experiments/logs/{args.name}"),
        model_path=f"experiments/models/{args.name}",
    )

    trainer.train_epochs(1000)


if __name__ == "__main__":
    main()