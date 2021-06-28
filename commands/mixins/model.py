from django.conf import settings

from systems.commands.index import CommandMixin
from utility.project import project_dir


class ModelMixin(CommandMixin('model')):

    @property
    def _model_project(self):
        return project_dir(settings.MODEL_PROJECT_NAME, settings.MODEL_PROJECT_MODEL_DIR)


    def get_model(self, name, model_params, data_params):
        model_params['data_provider'] = self.get_data_provider()

        if 'X_data' in model_params and isinstance(model_params['X_data'], str):
            index_column = model_params.pop('X_data_index_column') if 'X_data_index_column' in model_params else 'date'
            model_params['X_data'] = self.load_data(model_params['X_data'], index_column = index_column)

        if 'Y_data' in model_params and isinstance(model_params['Y_data'], str):
            index_column = model_params.pop('Y_data_index_column') if 'Y_data_index_column' in model_params else 'date'
            model_params['Y_data'] = self.load_data(model_params['Y_data'], index_column = index_column)

        return self.get_provider('ml_model', self.get_model_provider(),
            name, model_params, data_params
        )


    def get_model_provider(self):
        raise NotImplementedError("Implement get_model_provider in derived classes of the Model Command Mixin")

    def get_data_provider(self):
        raise NotImplementedError("Implement get_data_provider in derived classes of the Model Command Mixin")