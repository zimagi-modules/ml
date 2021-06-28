from tensorflow import keras
from django.conf import settings

from systems.commands.index import BaseCommand


class KerasBaseCommand(BaseCommand('keras')):

    @property
    def keras(self):
        return keras


    def get_model_provider(self):
        return 'keras'

    def get_data_provider(self):
        return 'keras'

    def data_parameters(self):
        return { **super().data_parameters(),
            'training_index': self.training_index,
            'validation_index': self.validation_index,
            'test_index': self.test_index
        }


    def train(self, model, params):
        self.notice('Training model')
        return model.train(
            epochs = self.epochs,
            batch_size = self.batch_size,
            verbose = not settings.API_EXEC,
            **params
        )
