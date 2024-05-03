from .training_config import TrainingConfig

from util import SuperResolutionDataset
from util.dataset_names import DIV2K_Training, DIV2K_Validation


class SuperResTrainingConfig(TrainingConfig):
    def __init__(self, batch_size: int, learning_rate: float, epochs: int = 100, loss='mse'):
        super().__init__(
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            training_set=SuperResolutionDataset(DIV2K_Training),
            validation_set=SuperResolutionDataset(DIV2K_Validation),
            loss=loss
        )
