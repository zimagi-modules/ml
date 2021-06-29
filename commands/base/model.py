from django.conf import settings

from systems.commands.index import BaseCommand


class ModelBaseCommand(BaseCommand('model')):

    def model_parameters(self):
        return {}

    def data_parameters(self):
        return {
            'split_percentages': self.split_percentages
        }

    def train_parameters(self):
        return {}

    def predict_parameters(self):
        return {}


    def exec(self):
        self.run_model()

    def run_model(self):
        model = self.init_model()
        training = self.train(model)
        test_data = self.predict(model)
        return model, training, test_data


    def init_model(self):
        model_params = self.model_parameters()
        model_params['X_data'] = self.predictor_data_name
        model_params['Y_data'] = self.target_data_name

        model = self.get_model(self.model_name,
            model_params, self.data_parameters()
        )
        self.notice('Building model')
        model.build(self.build_model)

        self.data('Model', self.model_name)
        self.info(model.summary())
        return model

    def build_model(self, data):
        # Override in subclass
        raise NotImplementedError("Subclasses of the base model command must implement a build_model method")


    def train(self, model):
        self.notice('Training model')
        return model.train(**self.train_parameters())

    def predict(self, model):
        self.notice('Running test predictions and generating results')
        test_data = model.predict(**self.predict_parameters())

        if self.plot_columns:
            suffixes = self.plot_prediction_suffixes if self.plot_prediction_suffixes else None

            self.notice('Generating test plots')
            for column in self.plot_columns:
                model.plot(column, test_data,
                    *model.get_prediction_columns(column, suffixes = suffixes)
                )
        self.success('Successfully tested and generated model results')
        return test_data
