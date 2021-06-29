from systems.plugins.index import BaseProvider
from utility.data import ensure_list


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

    def get_test_index(self):
        return self.test_data.index


    def get_min_samples(self):
        return 3


    def postprocess(self):
        self.training_frame = self.reframe(self.training_data)
        self.validation_frame = self.reframe(self.validation_data)
        self.test_frame = self.reframe(self.test_data)


    def reframe(self, data):
        results = numpy.array(data)
        return results.reshape((data.shape[0], data.shape[1]))


    def get_prediction_columns(self, column = None, suffixes = None):
        columns = list(self.test_data.columns)
        for test_column in self.test_data.columns:
            if (column is None or column == test_column) and (suffixes is None or '1' in ensure_list(suffixes)):
                columns.append("{}_1".format(test_column))
        return columns
