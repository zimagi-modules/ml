from tensorflow import keras
from matplotlib import pyplot

from systems.plugins.index import BaseProvider
from utility.data import ensure_list

import pandas


class Provider(BaseProvider("ml_model", "keras")):

    @property
    def prediction_columns(self):
        columns = list(self.Y.test_data.columns)
        for index in range(self.Y.period):
            columns.extend(["{}_{}".format(column, index + 1) for column in self.Y.test_data.columns])
        return columns

    def get_prediction_columns(self, column, include_column = True, indexes = None):
        columns = [column] if include_column else []
        for index in range(self.Y.period):
            if indexes is None or (index + 1) in ensure_list(indexes):
                columns.append("{}_{}".format(column, index + 1))
        return columns


    def load_model(self, model_path):
        return keras.models.load_model(model_path)

    def save_model(self, model_path):
        self.model.save(model_path)


    def summary(self):
        return str(self.model.summary()) if self.model else None


    def train(self, epochs = 10, batch_size = 32, **params):
        results = self.model.fit(
            self.X.training,
            self.Y.training,
            validation_data = (self.X.validation, self.Y.validation),
            epochs = epochs,
            batch_size = batch_size,
            **params
        )
        self.plot_loss(results)
        return results

    def predict(self, **params):
        predictions = self.model.predict(self.X.test)
        test_data = []

        for index, Y_info in enumerate(self.Y.test):
            # Next actual values only
            record = list(Y_info[0])

            # All prediction timeframes
            for prediction in predictions[index]:
                record.extend(prediction)

            test_data.append(record)

        series = pandas.DataFrame(test_data,
            columns = self.prediction_columns,
            index = self.Y.test_data.index[self.X.period:-(self.Y.period - 1)]
        )
        self.export('predictions', series)
        return series


    def plot_loss(self, results):
        with self._result_project as project:
            pyplot.title("Model {} loss".format(self.model_id))
            pyplot.ylabel('Loss')
            pyplot.xlabel('Epoch')

            pyplot.plot(results.history['loss'])
            pyplot.plot(results.history['val_loss'])

            pyplot.legend(['Train', 'Validation'], loc = 'upper left')
            pyplot.savefig("{}/{}_loss.png".format(project.base_path, self.model_id))
            pyplot.close()
