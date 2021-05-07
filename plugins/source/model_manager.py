# filename matches name given in plugins data definition
from systems.plugins.index import BaseProvider

import logging
import pandas as pd
import io

logger = logging.getLogger(__name__)

class Provider(BaseProvider("source", "model_manager")):
    # Generate a parent class based on 'source' and plugin definition
    # Three interface methods required: item_columns, load_items, load_item

    def __init__(self):
        self.columns = self.features + [self.target]

    def item_columns(self):
        # Return a list of header column names for source dataframe
        return self.columns

    def load_items(self, context):
        yield from self.facade(self.view).values(self.columns)

    def load_item(self, row, context):
        pass
