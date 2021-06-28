from systems.plugins.index import BasePlugin

import math


class BaseProvider(BasePlugin('ml_data')):

    def __init__(self, type, name, command, data, **config):
        super().__init__(type, name, command)
        self.data = self.preprocess(data)
        self.config = config

        assert len(self.field_split_percentages) >= self.get_min_samples() and sum(self.field_split_percentages) <= 1

        self.data_length = self.data.shape[0]
        self.feature_count = self.data.shape[1]
        self.samples = self.split_data(self.field_split_percentages)
        self.postprocess()


    def get_min_samples(self):
        return 2


    def preprocess(self, data):
        # Override in subclass if needed
        return data

    def postprocess(self):
        # Override in subclass if needed
        pass


    def split_data(self, percentages):
        sample_data = list()
        index = 0

        for percentage in percentages:
            length = math.floor(self.data_length * percentage)
            sample_data.append(self.normalize(self.data[index:length]))
            index += length

        return sample_data


    def normalize(self, data):
        return data.fillna(0)
