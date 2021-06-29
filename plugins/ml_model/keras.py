from tensorflow import keras
from matplotlib import pyplot

from systems.plugins.index import BaseProvider
from utility.data import ensure_list

import pandas


class Provider(BaseProvider("ml_model", "keras")):

    def load_model(self, model_path):
        return keras.models.load_model(model_path)

    def save_model(self, model_path):
        self.model.save(model_path)


    def summary(self):
        return str(self.model.summary()) if self.model else None


    def train(self, epochs = 10, batch_size = 32, **params):
        results = self.model.fit(
            self.X.training_frame,
            self.Y.training_frame,
            validation_data = (self.X.validation_frame, self.Y.validation_frame),
            epochs = epochs,
            batch_size = batch_size,
            **params
        )
        self.plot_loss(results)
        return results

    def predict(self, **params):
        predictions = self.model.predict(self.X.test_frame)
        Y_columns = list(self.Y.test_data.columns)
        test_data = []

        for index, Y_info in enumerate(self.Y.test_frame):
            # Next actual values only
            record = list(Y_info[0])

            # All prediction timeframes
            for prediction in list(predictions[index]):
                try:
                    record.extend(prediction)
                except Exception:
                    record.append(prediction)

            test_data.append(record)

        data = pandas.DataFrame(test_data,
            columns = self.Y.get_prediction_columns(),
            index = self.Y.get_test_index()
        )
        self.export('predictions', data)
        return data


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
