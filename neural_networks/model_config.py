import json


class ModelConfig:
    def __init__(self, name, quantised, bit_width=None, loss='mse'):
        self.name = name
        self.quantised = quantised
        self.bit_width = bit_width
        self.loss = loss

    def json(self) -> dict:
        dictionary = {
            "name": self.name,
            "quantised": self.quantised,
            "loss": self.loss
        }

        if self.quantised:
            dictionary["bit_width"] = self.bit_width

        return dictionary
