"""
Read raw draft data from a sqlite database and construct an analytic base table to use for model fitting.
"""
import sqlite3
import numpy as np
import pandas as pd

TABLE_NAMES = {'preferences', 'options', 'picks', 'cards'}


class AnalyticTableConstructor:

    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.validate_db()

    def validate_db(self):
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';")
        # We've got to unpack some singleton tuples here.
        table_names = {t[0] for t in self.cursor.fetchall()}
        assert table_names == TABLE_NAMES

    def make_analytic_base_table(self):
        options = self.fetch_table('options',
                                   table_validator=validate_options,
                                   name_transformer=name_sanitizer('options'))
        cards = self.fetch_table('cards',
                                 table_validator=validate_cards,
                                 name_transformer=name_sanitizer('cards'))
        picks = self.fetch_table('picks',
                                 table_validator=validate_picks)
        pick_idx = pd.Series(np.argmax(picks.values, axis=1), 
                             index=picks.index)
        joined = pd.merge(options, cards, left_index=True, right_index=True)
        joined["pick"] = pick_idx
        return joined

    def fetch_table(self, table_name, *,
                    table_validator=None,
                    name_transformer=None):
        # The table name cannot be passed in as a sanitized parameter, hence
        # the regretful string formatting here.
        table = pd.read_sql_query(f"select * from {table_name};", self.conn)
        table = table.set_index(['draft_id', 'drafter', 'pick_number'])
        if table_validator is not None:
            assert table_validator(table)
        if name_transformer is not None:
            table.columns = table.columns.map(name_transformer)
        return table


def name_sanitizer(prefix=''):
    def sanitize_name(name):
        return prefix + '_' + name.lower().replace(' ', '_')
    return sanitize_name

def validate_options(options):
    num_options = options.sum(axis=1)
    expected_num_options = (
        14 - options.index.get_level_values('pick_number').values % 14)
    return (num_options == expected_num_options).all()

def validate_cards(cards):
    num_cards = cards.sum(axis=1)
    expected_num_cards = cards.index.get_level_values('pick_number').values
    return (num_cards == expected_num_cards).all()

def validate_picks(picks):
    return (picks.sum(axis=1) == 1).all()
