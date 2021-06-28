from systems.plugins.index import BaseProvider


class Provider(BaseProvider("ml_data", "keras")):

    @property
    def training_data(self):
        return self.samples[self.field_training_index]

    @property
    def validation_data(self):
        return self.samples[self.field_validation_index]

    @property
    def test_data(self):
        return self.samples[self.field_test_index]


    def get_min_samples(self):
        return 3
