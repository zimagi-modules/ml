from systems.commands.index import Command


class Period(Command('dump.period')):

    def exec(self):
        self.save_data(self.data_name, self.get_combined_period(
            query_types = self.parse_query_fields(self.period_fields),
            start_time = self.period_start_time,
            end_time = self.period_end_time,
            unit_type = self.period_unit_type,
            units = self.period_units,
            required_types = self.required_types,
            last_known_value = not self.no_fill,
            resample = self.resample_freq,
            resample_summary = self.resample_summary
        ))
        self.success("Data {} successfully exported to file".format(self.data_name))


    def parse_query_fields(self, period_fields):
        query_types = {}

        for query_field, value in period_fields.items():
            query_type, field = query_field.split(':')

            if query_type not in query_types:
                query_types[query_type] = {}

            if ',' in value:
                value = value.split(',')

            query_types[query_type][field] = value

        return query_types
