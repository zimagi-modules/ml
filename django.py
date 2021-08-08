from settings.config import Config

#
# Model project configurations
#
MODEL_PROJECT_NAME = Config.string('ZIMAGI_MODEL_PROJECT_NAME', 'ml')
MODEL_PROJECT_MODEL_DIR = Config.string('ZIMAGI_MODEL_PROJECT_MODEL_DIR', 'models')
MODEL_PROJECT_RESULT_DIR = Config.string('ZIMAGI_MODEL_PROJECT_RESULT_DIR', 'results')
