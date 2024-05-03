from torch.utils.data import DataLoader


class TrainingConfig:
    def __init__(
            self,
            batch_size: int,
            learning_rate: float,
            training_set,
            validation_set,
            epochs: int = 100,
            loss: str = 'mse'
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.training_set = DataLoader(training_set, batch_size=batch_size, shuffle=True)
        self.validation_set = DataLoader(validation_set, batch_size=1, shuffle=False)
        self.loss = loss
