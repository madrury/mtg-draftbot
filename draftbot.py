import json
import random
import uuid
import sqlite3
import numpy as np
import pandas as pd

OPEN_ARCHYTYPE_SCALE = 2.0

class Draft:
    """Simulate a Magic: The Gathering draft with algorithmic drafters
    
    Parameters
    ----------
    n_drafters:
      The number of drafters in the draft pod.  Usually equal to 8.

    n_rounds:
      The number of rounds to the draft.  Usually equal to 3.

    deck_archetypes:
      Tuple contiaining names for the deck archetypes in the format being
      drafted.

    cards_path:
      A path to a json file containing card definitions for the given set.
      This data is aquired from mtgjson.

    card_values_path:
      A path to a json file contining ratings of each card in a set for each
      deck archytype.

    Attributes
    ----------
    cards:
      The deserialized json from cards_path.

    card_values:
      The deserialized json from card_values_path.

    set:
      An object representing the set being drafted.  Contains methods for
      randomizing a pack.

    drafters:
      A list of Drafter objects, representing each drafter participating.
    """
    def __init__(self, *, 
                 n_drafters=8,
                 n_rounds=3,
                 n_cards_in_pack=14,
                 cards_path=None,
                 card_values_path=None):
        self.draft_id = uuid.uuid4()
        self.n_drafters = n_drafters
        self.n_rounds = n_rounds
        self.n_cards_in_pack = n_cards_in_pack
        self.archetype_names = None # <+- Will be set in below method...
        self.card_names = None      #  |- Will be set in below method...
        # This is what a machine lerning model could learn from draft data.
        self.archetype_weights = self.make_archetype_weights_array(
            json.load(open(card_values_path)))
        self.n_archetypes = len(self.archetype_names)
        self.set = Set(cards=json.load(open(cards_path)), 
                       card_names=self.card_names)
        # Internal algorithmic data structure.
        self.drafter_preferences = np.ones(shape=(self.n_drafters, self.n_archetypes))
        self.round = 0
        # Output data structures.
        self.picks = np.zeros(
            (self.n_drafters, self.set.n_cards, self.n_cards_in_pack * self.n_rounds),
            dtype=int)
        self.preferences_history = np.zeros(
            (self.n_drafters, self.n_archetypes, self.n_cards_in_pack * self.n_rounds))
    
    def draft(self):
        for _ in range(self.n_rounds):
            self.draft_pack()
            self.round += 1

    def draft_pack(self):
        packs = self.set.random_packs_array(self.n_drafters)
        for n_pick in range(14):
            card_is_in_pack = np.sign(packs)
            pack_archetype_weights = (
                card_is_in_pack.reshape((self.n_drafters, self.set.n_cards, 1)) * 
                self.archetype_weights.reshape((1, self.set.n_cards, self.n_archetypes)))
            preferences = np.einsum(
                'dca,da->dc',pack_archetype_weights, self.drafter_preferences)
            pick_probs = softmax(preferences)
            picks = self.make_picks(pick_probs)
            # Update internal data structures
            packs = rotate_array(packs - picks, forward=True)
            self.drafter_preferences = (
                self.drafter_preferences +
                np.einsum('ca,pc->pa', self.archetype_weights, picks))
            # Update output data structures
            self.picks[:, :, n_pick + self.n_cards_in_pack * self.round] = picks.copy()
            self.preferences_history[:, :, n_pick + self.n_cards_in_pack * self.round] = (
                self.drafter_preferences.copy())

    def make_picks(self, pick_probs):
        picks = np.zeros((self.n_drafters, self.set.n_cards), dtype=int)
        for ridx, row in enumerate(pick_probs):
            pick_idx = np.random.choice(self.set.n_cards, p=row)
            picks[ridx, pick_idx] = 1
        return picks

    def make_archetype_weights_array(self, card_values):
        archetype_weights_df = pd.DataFrame(card_values).T
        self.archetype_names = archetype_weights_df.columns
        self.card_names = archetype_weights_df.index
        return archetype_weights_df.values

    def write_to_database(self, path):
        conn = sqlite3.connect(path)
        self._write_preferences_to_database(conn)
        self._write_picks_to_database(conn)
        conn.close()

    def _write_preferences_to_database(self, conn, if_exists='append'):
        for idx, ph in enumerate(self.preferences_history):
            df = pd.DataFrame(ph.T, columns=self.archetype_names)
            df['draft_id'] = str(self.draft_id)
            df['drafter'] = idx
            df['pick_number'] = np.arange(df.shape[0])
            df.to_sql("preferences", conn, index=False, if_exists=if_exists)

    def _write_picks_to_database(self, conn, if_exists='append'):
        for idx, picks in enumerate(self.picks):
            df = pd.DataFrame(picks.T, columns=self.card_names)
            df['draft_id'] = str(self.draft_id)
            df['drafter'] = idx
            df['pick_number'] = np.arange(df.shape[0])
            df.to_sql("picks", conn, index=False, if_exists='append')


class Set:

    def __init__(self, cards, card_names):
        self.cards = cards
        self.commons, self.uncommons, self.rares = self.split_by_rarity(cards)
        self.card_names = card_names
        self.n_cards = len(self.card_names)

    def random_packs_array(self, n_packs=8, pack_size=14):
        packs = [self.random_pack_dict(size=pack_size) for _ in range(n_packs)]
        cards_in_pack_df = pd.DataFrame(np.zeros(shape=(n_packs, self.n_cards), dtype=int), 
                                        columns=self.card_names)
        for idx, pack in enumerate(packs):
            for card in pack:
                name = card['name']
                cards_in_pack_df.loc[cards_in_pack_df.index[idx], name] += 1
        return cards_in_pack_df.values 
    
    def random_pack_dict(self, size=14):
        n_rares, n_uncommons, n_commons = 1, 3, size - 4
        pack = []
        for _ in range(n_commons):
            pack.append(random.choice(self.commons))
        for _ in range(n_uncommons):
            pack.append(random.choice(self.uncommons))
        pack.append(random.choice(self.rares))
        return pack

    @staticmethod
    def split_by_rarity(cards):
        commons, uncommons, rares = [], [], []
        for card in cards:
            rarity = card['rarity']
            if rarity == 'common':
                commons.append(card)
            elif rarity == 'uncommon':
                uncommons.append(card)
            elif rarity in {'rare', 'mythic'}:
                rares.append(card)
        return commons, uncommons, rares


def softmax(x):
    exps = np.exp(x)
    row_sums = np.sum(exps, axis=1)
    probs = exps / row_sums.reshape(-1, 1)
    return probs

def rotate_array(x, forward=True):
    newx = np.zeros(x.shape)
    if forward:
        newx[0, :] = x[-1, :]
        newx[1:, :] = x[:-1, :]
    else:
        newx[-1, :] = x[0, :]
        newx[:-1, :] = x[1:, :]
    return newx

# Old bells and whistles, may use these again in the future.
def convert_to_discount_factor(pick_num, pack_size=14):
    return 1 / (1 + np.exp((pick_num % pack_size - 8)))

def make_sigmoid(*, a=2, b=1, t0=0, rate=1):
    def sigmoid(t):
        return (b - a) / (1 + np.exp(- rate * (t - t0))) + a
    return sigmoid

zero_color_sigmoid = make_sigmoid(a=4/10, b=1, t0=3, rate=2) 
multi_color_sigmoid = make_sigmoid(a=7/10, b=1, t0=3, rate=2)

def normalize_array(x, scale=1.0):
     return 2 * scale * ((x - x.min()) / (x.max() - x.min()) - 0.5)
