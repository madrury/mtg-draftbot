"""
Read raw draft data from a sqlite database and construct an analytic base table to use for model fitting.
"""
import sqlite3
import numpy as np
import pandas as pd

TABLE_NAMES = {'preferences', 'options', 'picks', 'cards'}


class AnalyticTableConstructor:
    """Construct an analytic base table for learning draft archetype weights
    from draft pick data.

    Parameters
    ----------
    db_path: Union[str, Path]
      Path to an sqlite3 data base containing draft data.

    Given this database, the make_analytic_base_table method constructs
    analytic style data useful for machine learning.
    """
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.validate_db()

    def make_analytic_base_table(self):
        """Construct analytic machine learning data from the tables in the
        database.

        Returns
        -------
        X: pd.DataFrame[int]
          Data frame indexed by (draft_id, drafter_id, pick_number) containing
          features known at the time of a given pick in a draft, intended to be
          used as predictors for the next pick. There predictors come in two types:
            - The options for picks from the current pack.
            - The cards held by the current drafter (from past picks).
        
        y: pd.Series[int]
          Data series indexed by (draft_id, drafter_id, pick_number) containing
          the pick made by the drafter, intended to be a used as a target for a
          predictive model.

        y_to_name_mapping: Dict[int, str]
          Dictionary mapping the interger id of a card used in the y series to
          the actual name of the card.
        """
        options = self.fetch_table('options',
                                   table_validator=validate_options,
                                   name_transformer=name_sanitizer('options'))
        cards = self.fetch_table('cards',
                                 table_validator=validate_cards,
                                 name_transformer=name_sanitizer('cards'))
        X = pd.merge(options, cards, left_index=True, right_index=True)
        picks = self.fetch_table('picks',
                                 table_validator=validate_picks)
        y = pd.Series(np.argmax(picks.values, axis=1), 
                          index=picks.index)
        y_to_name_mapping = dict(enumerate(picks.columns))
        return X, y, y_to_name_mapping

    def validate_db(self):
        """Make sure that the database contains the needed tables."""
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';")
        # We've got to unpack some singleton tuples here.
        table_names = {t[0] for t in self.cursor.fetchall()}
        assert table_names == TABLE_NAMES

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
