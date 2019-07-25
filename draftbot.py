import json
import random
import uuid
import sqlite3
import numpy as np
import pandas as pd


class Draft:
    """Simulate a Magic: The Gathering draft with algorithmic drafters.

    Parameters
    ----------
    n_drafters: int
      The number of drafters in the draft pod.  Usually equal to 8.

    n_rounds: int
      The number of rounds to the draft.  Usually equal to 3.

    n_cards_in_pack: int
      The number of cards in a single pack, equal to the number of picks a
      player must make in one round of the draft.

    cards_path:
      A path to a json file containing card definitions for the given set.
      This data is aquired from mtgjson.

    card_values_path:
      A path to a json file contining ratings of each card in a set for each
      deck archytype.

    Description of The Algorithm
    ----------------------------
    This algorithm depends on a prior enumeration of the various deck
    archetypes in the draft format (though, the actual identities of these
    could be learned from actual human draft data, in which case only the
    number of such archetypes would need to be pre-specified).

    Given this enumeration of archetypes, there are two essential data
    structures used in this algorithm:

    drafter_preferences: np.array, shape (n_drafters, n_archetypes)
      This array tracks the internal preference of each drafter for each
      archetype.  This array contributes linearly to the relative log-odds that
      a card in any given pack will be picked, and is updated each time a pick
      is made (further preferencing the archetypes for which the selected card
      is valuable).

    archetype_weights: np.array, shape (n_cards, n_archetypes)
      This array contians relative ratings of how valuable each card is in each
      archetypes.  Throughout a simulated draft, this array is static.  This
      array could concevably be learned from human draft data, and then used to
      simulate algorithmic drafters.

    Given a single drafter considering a single pick from some number of
    available cards, the preference of the drafter for each card is computed as
    the dot product of their archetype preferences with the archetype weights
    for that card in each available archetype.  These preferences are converted
    into probabilities using a softmax, and then a card is selected using this
    distribution.

    After a card is selected, the drafter's preferences are updated by adding
    the archetype weights for the selected card tothe drafter's current
    archetype preferences.

    Output Attributes
    -----------------
    Certain object attributes are available as output, and contiain complete
    information about the progress of the draft.

    options: np.array, shape (n_drafters, n_cards, n_cards_in_pack * n_rounds)
      The options avaailable for each drafter over each pick of the draft.
      Entries in this array are counts of how many of each card is available to
      the given drafter over each pick of the draft.

    picks: np.array, shape (n_drafters, n_cards, n_cards_in_pack * n_rounds)
      Which card is chosen by each drafter over each pick of the draft.
      Entries in this array are either zero or one, and there is a single one
      in each 1-dimensional slice of the array of the form [d, :, p].

    cards: np.array, shape (n_drafters, n_cards, n_cards_in_pack * n_rounds)
      The current set of cards owned by each drafter at each pick of the draft.
      Equal to the cumlative sum of self.picks over the final axis, shifted up
      one index (since there are no cards owned by any player for the first
      pick of the draft).

    preferences: np.array, shape (n_drafters, n_archetypes, n_cards_in_pack * n_rounds)
      The preferences of each drafter for each archetype over each pick of the
      draft.  Each 1-dimensional slice of this array of the form [d, :, p]
      contains the current preferences for a drafter at a single pick of the
      draft.
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
        # These archetype weights could be learned from human draft data.
        self.archetype_weights, self.archetype_names, self.card_names = (
            self.make_archetype_weights_array(json.load(open(card_values_path))))
        self.n_archetypes = len(self.archetype_names)
        self.set = Set(
            cards=json.load(open(cards_path)),
            card_names=self.card_names)
        # Internal algorithmic data structure.
        self.drafter_preferences = np.ones(shape=(self.n_drafters, self.n_archetypes))
        self.round = 0
        # Output data structures.
        self.options = np.zeros(
            (self.n_drafters, self.set.n_cards, self.n_cards_in_pack * self.n_rounds),
            dtype=int)
        self.picks = np.zeros(
            (self.n_drafters, self.set.n_cards, self.n_cards_in_pack * self.n_rounds),
            dtype=int)
        self.preferences = np.zeros(
            (self.n_drafters, self.n_archetypes, self.n_cards_in_pack * self.n_rounds))

    def draft(self):
        for _ in range(self.n_rounds):
            packs = self.set.random_packs_array(n_packs=self.n_drafters)
            for n_pick in range(self.n_cards_in_pack):
                packs = self.draft_packs(packs, n_pick)
            self.round += 1

    def draft_packs(self, packs, n_pick):
        """Draft a single pick from a set of packs, one pick for each drafter.

        Parameters
        ----------
        packs: np.array, shape (n_drafters, n_cards)
          Array indicating how many of each card remains in each pack.

        n_pick: int
          Which pick of the draft is this?  Note that the first pick of the
          draft is pick zero.

        Modifies
        --------
        self.options: np.array 
                      shape (n_drafters, n_cards, n_cards_in_pack * n_rounds)
          The initial packs array (before any picks are made) is copied into an
          (n_drafters, n_cards) slice of this output array.

        self.picks: np.array
                    shape (n_drafters, n_cards, n_cards_in_pack * n_rounds)
          After picks are computed for each drafter, these are copied into an
          (n_drafters, n_cards) slice of this output array.

        self.preferences: np.array
                          shape (n_drafters, n_archetypes, n_cards_in_pack * n_rounds)
          After archytype preferences are re-computed for each drafter post
          pick, these are copied into an (n_drafters, n_archetypes) slice of
          this output array.

        Returns
        -------
        packs: np.array, shape (n_drafters, n_cards)
          The input packs array, transformed given the results of the pick algorithm:
            - The cards picked by each drafter are removed from the corresponding packs.
            - The packs array is rotated (in the first dimension), to prepare
              for the next pick (packs are passed around the drafters
              circularly pick-to-pick.
        """
        self.options[:, :, n_pick + self.n_cards_in_pack * self.round] = packs.copy()
        card_is_in_pack = np.sign(packs)
        pack_archetype_weights = (
            card_is_in_pack.reshape((self.n_drafters, self.set.n_cards, 1)) *
            self.archetype_weights.reshape((1, self.set.n_cards, self.n_archetypes)))
        preferences = np.einsum(
            'dca,da->dc', pack_archetype_weights, self.drafter_preferences)
        pick_probs = non_zero_softmax(preferences)
        picks = self.make_picks(pick_probs)
        # We should not be able to pick a card that does not exist.
        assert np.all(packs >= picks)
        packs = rotate_array(packs - picks, forward=True)
        self.drafter_preferences = (
            self.drafter_preferences +
            np.einsum('ca,dc->da', self.archetype_weights, picks))
        self.picks[:, :, n_pick + self.n_cards_in_pack * self.round] = picks.copy()
        self.preferences[:, :, n_pick + self.n_cards_in_pack * self.round] = (
            self.drafter_preferences.copy())
        return packs

    def make_picks(self, pick_probs):
        """Compute definate picks given an array of pick probabilities.

        Parameters
        ----------
        pick_probs: np.array, shape (n_drafters, n_cards)
          Pick probabilities for each drafter.

        Returns
        -------
        picks: np.array, shape (n_drafters, n_cards)
          Definate picks for each drafter, computed by sampling from according
          to the each row of the pick probability array. 
        """
        picks = np.zeros((self.n_drafters, self.set.n_cards), dtype=int)
        for ridx, row in enumerate(pick_probs):
            #pick_idx = np.random.choice(self.set.n_cards, p=row)
            pick_idx = np.argmax(row)
            picks[ridx, pick_idx] = 1
        return picks

    @property
    def cards(self):
        cards = np.zeros(shape=self.picks.shape, dtype=int)
        cards[:, :, 1:] = np.cumsum(self.picks, axis=2)[:, :, 0:-1]
        return cards

    def make_archetype_weights_array(self, card_values):
        archetype_weights_df = pd.DataFrame(card_values).T
        archetype_names = archetype_weights_df.columns
        card_names = archetype_weights_df.index
        return archetype_weights_df.values, archetype_names, card_names

    def write_to_database(self, path):
        conn = sqlite3.connect(path)
        self._write_preferences_to_database(conn)
        self._write_options_to_database(conn)
        self._write_picks_to_database(conn)
        self._write_cards_to_database(conn)
        conn.close()

    def _write_array_to_database(self, conn, *,
                                 array,
                                 column_names,
                                 table_name,
                                 if_exists='append'):
        for idx, table in enumerate(array):
            df = pd.DataFrame(table.T, columns=column_names)
            df['draft_id'] = str(self.draft_id)
            df['drafter'] = idx
            df['pick_number'] = np.arange(df.shape[0])
            df.to_sql(table_name, conn, index=False, if_exists=if_exists)

    def _write_preferences_to_database(self, conn, if_exists='append'):
        self._write_array_to_database(conn,
                                      array=self.preferences,
                                      column_names=self.archetype_names,
                                      table_name="preferences")

    def _write_options_to_database(self, conn, if_exists='append'):
        self._write_array_to_database(conn,
                                      array=self.options,
                                      column_names=self.card_names,
                                      table_name="options")

    def _write_picks_to_database(self, conn, if_exists="append"):
        self._write_array_to_database(conn,
                                      array=self.picks,
                                      column_names=self.card_names,
                                      table_name="picks")

    def _write_cards_to_database(self, conn, if_exists="append"):
        self._write_array_to_database(conn,
                                      array=self.cards,
                                      column_names=self.card_names,
                                      table_name="cards")


class Set:
    """A set of cards to draft from.

    Parameters
    ----------
    cards: List[Dict[str, object]]
      Data representing all the cards in the set, created by processing data
      from mtgjson.com. Each card is represented by a dictionary inside this
      list. Important keys in the dictionary are:
        - name: The name of the card.
        - rarity: The rarity of the card, common, uncommon, rare, or mythic.
        - colorIdentity: The colors of the card.

    card_names: List[str]
      A list of card names in the set. This is used here to create a canonical
      ordering of the cards in the set, since different sources may be
      inconsistent about the ordering.
    """
    def __init__(self, cards, card_names):
        self.cards = cards
        self.commons, self.uncommons, self.rares = self.split_by_rarity(cards)
        self.card_names = card_names
        self.n_cards = len(self.card_names)

    def random_packs_array(self, n_packs=8, pack_size=14):
        """Make some random packs of cards from the set. Each pack contains one
        rare/mytic, thre uncommons, and however many commons are needed to
        round out the pack.

        Parameters
        ----------
        n_packs: int
          The number of packs to create.

        pack_size: int
          The number of cards in each pack.

        Returns
        -------
        packs: np.array, shape (n_packs, n_cards)
          The number of each card in each pack.
        """
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


def non_zero_softmax(x):
    x = x / x.max()  # For numerical reasons, to avoid exps of massive numbers.
    non_zero_exps = (x > 0) * np.exp(x)
    row_sums = np.sum(non_zero_exps, axis=1)
    probs = non_zero_exps / row_sums.reshape(-1, 1)
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
