from django.conf import settings

from systems.commands.index import CommandMixin
from utility.time import Time
from utility.data import ensure_list
from utility.query import init_fields, init_filters
from utility.project import project_dir
from utility.dataframe import merge, get_csv_file_name

import pandas


class DataProcessingMixin(CommandMixin('data_processing')):

    def __init__(self, name, parent = None):
        super().__init__(name, parent)

        self.time_processor = Time(
            date_format = settings.MODEL_DEFAULT_DATE_FORMAT,
            time_format = settings.MODEL_DEFAULT_TIME_FORMAT,
            spacer = settings.MODEL_DEFAULT_TIME_SPACER_FORMAT
        )


    @property
    def _data_project(self):
        return project_dir(settings.MODEL_PROJECT_NAME, settings.MODEL_PROJECT_DATA_DIR)

    @property
    def _result_project(self):
        return project_dir(settings.MODEL_PROJECT_NAME, settings.MODEL_PROJECT_RESULT_DIR)


    def save_data(self, name, data, project_context = None, time_processor = None, **options):
        if not project_context:
            project_context = self._data_project

        if not time_processor:
            time_processor = self.time_processor

        with project_context as project:
            project.save(
                data.to_csv(date_format = time_processor.time_format, **options),
                get_csv_file_name(name)
            )

    def load_data(self, name, project_context = None, index_column = 'date', sort_index = True, **options):
        if not project_context:
            project_context = self._data_project

        with project_context as project:
            data = pandas.read_csv(project.path(get_csv_file_name(name)), index_col = index_column, **options)

            if sort_index:
                data = data.sort_index()

        return data


    def get_data_record(self, data_type, time,
        fields = None,
        filters = None,
        index_field = 'date',
        order = '-date',
        recent = False
    ):
        filters = init_filters(filters)

        if recent:
            filters["{}__lte".format(index_field)] = time
        else:
            filters[index_field] = time

        return self.get_data_item(data_type, *init_fields(fields),
            filters = filters,
            order = order,
            dataframe = True,
            dataframe_index_field = index_field
        )

    def get_data_period(self, data_type,
        start_time = None,
        unit_type = 'days',
        units = None,
        fields = None,
        filters = None,
        index_field = 'date',
        order = 'date',
        resample = None,
        resample_summary = 'last',
        time_processor = None
    ):
        filters = init_filters(filters)

        if not time_processor:
            time_processor = self.time_processor

        if start_time and units:
            times = [start_time, time_processor.shift(start_time, units,
                unit_type = unit_type,
                to_string = True
            )]
            filters["{}__range".format(index_field)] = sorted(times)

        data = self.get_data_set(data_type, *init_fields(fields),
            filters = filters,
            order = order,
            dataframe = True,
            dataframe_index_field = index_field
        )
        if resample:
            data = getattr(data.resample(resample), resample_summary)()

        return data


    def get_combined_period(self, query_types,
        start_time = None,
        end_time = None,
        unit_type = 'days',
        units = None,
        required_types = None,
        last_known_value = True,
        resample = None,
        resample_summary = 'last',
        time_processor = None
    ):
        required_types = ensure_list(required_types) if required_types else None
        required_columns = list()
        periods = list()

        if not time_processor:
            time_processor = self.time_processor

        if start_time:
            if end_time:
                units = time_processor.distance(start_time, end_time,
                    unit_type = unit_type,
                    include_direction = True
                )
            elif not units:
                units = time_processor.distance(start_time, time_processor.now,
                    unit_type = unit_type,
                    include_direction = True
                )

        for query_type, params in query_types.items():
            data = getattr(self, "get_{}_period".format(query_type))(
                start_time = start_time,
                unit_type = unit_type,
                units = units,
                last_known_value = last_known_value,
                time_processor = time_processor,
                **params
            )
            if not required_types or query_type in required_types:
                required_columns.extend(list(data.columns))

            periods.append(data)

        data = merge(*periods,
            required_fields = required_columns,
            ffill = last_known_value
        )
        if resample:
            data = getattr(data.resample(resample), resample_summary)()

        return data
