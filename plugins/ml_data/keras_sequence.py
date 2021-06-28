from systems.plugins.index import BaseProvider

import numpy


class Provider(BaseProvider("ml_data", "keras_sequence")):

    def postprocess(self):
        self.training = self.reframe(self.training_data)
        self.validation = self.reframe(self.validation_data)
        self.test = self.reframe(self.test_data)
        self.period = self.field_Y_period if self.field_target else self.field_X_period


    def reframe(self, series):
        feature_count = series.shape[1]
        series = series.values
        results = list()

        for window_start in range(len(series)):
            X_end = window_start + self.field_X_period
            Y_end = X_end + self.field_Y_period

            if Y_end > len(series):
                break

            if self.field_target:
                results.append(series[X_end:Y_end,:])
            else:
                results.append(series[window_start:X_end,:])

        results = numpy.array(results)
        return results.reshape((results.shape[0], results.shape[1], feature_count))
