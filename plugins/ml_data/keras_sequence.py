from systems.plugins.index import BaseProvider
from utility.data import ensure_list, intersection

import copy
import numpy


class Provider(BaseProvider("ml_data", "keras_sequence")):

    def get_test_index(self):
        return self.test_data.index[self.field_X_period:(self.field_X_period + len(self.test_frame))]


    def postprocess(self):
        super().postprocess()

        if self.field_target:
            self.period = 1 if self.field_single_target else self.field_Y_period
        else:
            self.period = self.field_X_period


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
                if self.field_single_target:
                    results.append(series[(Y_end - 1):Y_end,:])
                else:
                    results.append(series[X_end:Y_end,:])
            else:
                results.append(series[window_start:X_end,:])

        results = numpy.array(results)
        return results.reshape((results.shape[0], results.shape[1], feature_count))


    def get_prediction_columns(self, column = None, suffixes = None):
        columns = intersection(list(self.test_data.columns), column, True)
        for index in range(self.period):
            if suffixes is None or str(index + 1) in ensure_list(suffixes):
                for index_column in intersection(list(self.test_data.columns), column, True):
                    columns.append("{}_{}".format(index_column, index + 1))
        return columns
