import torch
from pydantic import BaseModel


class Config(BaseModel):
    hidden_state_dim: int
    num_lstm_layers: int
    prediction_sequence_length: int


def model_from_config(config: Config, feature_dim: int) -> torch.nn.Module:
    return TrajectoryPredictorLSTM(
        input_features_dim=feature_dim,
        hidden_state_dim=config.hidden_state_dim,
        output_features_dim=feature_dim,
        num_lstm_layers=config.num_lstm_layers,
        prediction_sequence_length=config.prediction_sequence_length,
    )


def read_config(file_path: str) -> Config:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = f.read()
    return Config.model_validate_json(raw_data)


class TrajectoryPredictorLSTM(torch.nn.Module):
    def __init__(
        self,
        input_features_dim: int,
        hidden_state_dim: int,
        output_features_dim: int,
        num_lstm_layers: int,
        prediction_sequence_length: int,
    ):
        super().__init__()

        self.hidden_state_dim = hidden_state_dim
        self.num_lstm_layers = num_lstm_layers
        self.prediction_sequence_length = prediction_sequence_length

        self.encoder = torch.nn.LSTM(
            input_size=input_features_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        self.decoder = torch.nn.LSTM(
            input_size=hidden_state_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        self.projection = torch.nn.Linear(hidden_state_dim, output_features_dim)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        batch_size = input_sequence.size(0)
        device = input_sequence.device

        _, (h_n, c_n) = self.encoder(input_sequence)

        decoder_input = torch.zeros(
            batch_size,
            self.prediction_sequence_length,
            self.hidden_state_dim,
            device=device,
        )

        decoder_output, _ = self.decoder(decoder_input, (h_n, c_n))
        output = self.projection(decoder_output)

        return output