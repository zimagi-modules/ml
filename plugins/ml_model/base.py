from django.conf import settings
from matplotlib import pyplot

from systems.plugins.index import BasePlugin
from utility.datetime import DateTime
from utility.project import project_dir
from utility.dataframe import get_csv_file_name

import numpy


class BaseProvider(BasePlugin('ml_model')):

    def __init__(self, type, name, command, config):
        super().__init__(type, name, command)
        self.config = config

        if self.field_date_time is None:
            self.config['date_time'] = DateTime(
                settings.MODEL_DEFAULT_DATE_FORMAT,
                settings.MODEL_DEFAULT_TIME_FORMAT,
                settings.MODEL_DEFAULT_TIME_SPACER_FORMAT
            )

        self.model = None
        self.timestamp = self.field_date_time.now

        if self.field_X_data is not None:
            self.set_data(
                self.field_X_data,
                self.field_Y_data
            )


    @property
    def model_id(self):
        return "{}-{}".format(self.field_model_name, self.timestamp)


    @property
    def _data_project(self):
        return project_dir(settings.MODEL_PROJECT_NAME, settings.MODEL_PROJECT_DATA_DIR)

    @property
    def _model_project(self):
        return project_dir(settings.MODEL_PROJECT_NAME, settings.MODEL_PROJECT_MODEL_DIR)

    @property
    def _result_project(self):
        return project_dir(settings.MODEL_PROJECT_NAME, settings.MODEL_PROJECT_RESULT_DIR)


    def set_data(self, X_data, Y_data = None, **config):
        if Y_data is None:
            Y_data = X_data

        self.config['X_data'] = X_data
        self.config['Y_data'] = Y_data

        shared_indexes = numpy.intersect1d(X_data.index, Y_data.index)

        self.X = self.command.get_provider('ml_data', self.field_data_provider,
            X_data.loc[shared_indexes],
            config
        )
        self.Y = self.command.get_provider('ml_data', self.field_data_provider,
            Y_data.loc[shared_indexes],
            { **config, **{ 'target': True } }
        )


    def load(self, model_builder, *method_args, **method_kwargs):
        with self._model_project as project:
            model_path = project.path(self.model_id)

            if project.exists(self.model_id):
                self.model = self.load_model(model_path)
            else:
                self.build(model_builder, *method_args, **method_kwargs)
        return self.model

    def build(self, model_builder, *method_args, **method_kwargs):
        with self._model_project as project:
            self.model = model_builder(self, *method_args, **method_kwargs)
            self.save()
        return self.model

    def load_model(self, model_path):
        raise NotImplementedError("Implement load_model in derived classes of the base Machine Learning Model provider")


    def save(self):
        with self._model_project as project:
            self.save_model(project.path(self.model_id))

    def save_model(self, model_path):
        raise NotImplementedError("Implement save_model in derived classes of the base Machine Learning Model provider")


    def summary(self):
        raise NotImplementedError("Implement summary in derived classes of the base Machine Learning Model provider")


    def train(self, **params):
        raise NotImplementedError("Implement train in derived classes of the base Machine Learning Model provider")

    def predict(self, **params):
        raise NotImplementedError("Implement train in derived classes of the base Machine Learning Model provider")


    def export(self, name, data, **options):
        with self._result_project as project:
            project.save(
                data.to_csv(date_format = self.field_date_time.time_format, **options),
                get_csv_file_name("{}_{}".format(self.model_id, name))
            )


    def plot(self, name, data, *columns):
        pyplot.title("Model {} data: {}".format(self.model_id, name))
        pyplot.ylabel('Value')
        pyplot.xlabel('Samples')

        with self._result_project as project:
            for column in columns:
                pyplot.plot(data.loc[:,column])

            pyplot.legend(columns, loc = 'best')
            pyplot.savefig("{}/{}_model_{}.png".format(project.base_path, self.model_id, name))
            pyplot.close()
