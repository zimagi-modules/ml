from django.conf import settings

from systems.commands.index import CommandMixin
from utility.project import project_dir


class ModelMixin(CommandMixin('model')):

    @property
    def _model_project(self):
        return project_dir(settings.MODEL_PROJECT_NAME, settings.MODEL_PROJECT_MODEL_DIR)


    def get_model(self, name, **params):
        if 'X_data' in params and isinstance(params['X_data'], str):
            params['X_data'] = self.load_data(params['X_data'])

        if 'Y_data' in params and isinstance(params['Y_data'], str):
            params['Y_data'] = self.load_data(params['Y_data'])

        return self.get_model_class()(name, **params)


    def get_model_class(self):
        raise NotImplementedError("Implement get_model_class in derived classes of the Model Command Mixin")