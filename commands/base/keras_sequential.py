from systems.commands.index import BaseCommand


class KerasSequentialBaseCommand(BaseCommand('keras_sequential')):

    def get_data_provider(self):
        return 'keras_sequence'

    def data_parameters(self):
        return { **super().data_parameters(),
            'X_period': self.predictor_period,
            'Y_period': self.target_period
        }
