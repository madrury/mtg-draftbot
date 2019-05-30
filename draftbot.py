import json
import random
from math import exp
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
                 deck_archetypes=None,
                 cards_path=None,
                 card_values_path=None):
        self.n_drafters = n_drafters
        self.n_rounds = n_rounds
        self.deck_archetypes = deck_archetypes
        self.n_archetypes = len(self.deck_archetypes)
        self.cards = json.load(open(cards_path))
        self.n_cards = len(self.cards)
        self.card_values = json.load(open(card_values_path))
        self.set = Set(cards=self.cards, card_values=self.card_values)
        # This is what a machine lerning model could learn from draft data.
        self.archetype_weights = self.make_archetype_weights_array(
            self.card_values, self.deck_archetypes)
        # Internal algorithmic data structure.
        self.drafter_preferences = np.ones(shape=(self.n_drafters, self.n_archetypes))
        self.round = 0
        # Output data structures.
        self.picks = np.zeros(
            (self.n_drafters, self.n_cards, 14 * self.n_rounds))
        self.preferences_history = np.zeros(
            (self.n_drafters, self.n_archetypes, 14 * self.n_rounds))
    
    def draft(self):
        for _ in range(self.n_rounds):
            self.draft_pack()
            self.round += 1

    def draft_pack(self):
        packs = self.set.random_packs_array(self.n_drafters)
        for n_pick in range(14):
            card_is_in_pack = np.sign(packs)
            pack_archetype_weights = (
                card_is_in_pack.reshape((self.n_drafters, self.n_cards, 1)) * 
                self.archetype_weights.reshape((1, self.n_cards, self.n_archetypes)))
            preferences = np.einsum(
                'dca,da->dc',pack_archetype_weights, self.drafter_preferences)
            pick_probs = softmax(preferences)
            picks = self.make_picks(pick_probs)
            # Update internal data structures
            packs = packs - picks
            self.drafter_preferences = (
                self.drafter_preferences +
                np.einsum('ca,pc->pa', self.archetype_weights, picks))
            # Update output data structures
            self.picks[:, :, n_pick + 14 * self.round] = picks.copy()
            self.preferences_history[:, :, n_pick + 14 * self.round] = (
                self.drafter_preferences.copy())

    def make_picks(self, pick_probs):
        picks = np.zeros((self.n_drafters, self.n_cards), dtype=int)
        for ridx, row in enumerate(pick_probs):
            pick_idx = np.random.choice(self.n_cards, p=row)
            picks[ridx, pick_idx] = 1
        return picks

    def make_archetype_weights_array(self, card_values, deck_archetypes):
        archetype_weights_df = pd.DataFrame(card_values).T
        archetype_weights_df.columns = deck_archetypes
        return archetype_weights_df.values

#class Drafter:
#    """Represents a single drafter, and tracks that drafter's preferences
#    throughout the draft.
#
#    Parameters
#    ----------
#    """
#
#    def __init__(self, *, parent, archytype_preferences=None):
#        self.parent = parent
#        if not archytype_preferences:
#            archytype_preferences = {arch: 1 for arch in self.parent.deck_archetypes}
#        self.archytype_preferences = archytype_preferences
#        self.archytype_power_seen = {arch: 0 for arch in self.parent.deck_archetypes}
#        self.cards = []
#        self.archytype_preferences_history = [archytype_preferences.copy()]
#        self.archytype_power_seen_history = []
#
#    def pick(self, pack, pick_num):
#        self._update_power_seen(pack)
#        pick_scores = [self._calculate_score(card) for card in pack]
#        pick_probabilities = convert_to_probabilities_softmax(pick_scores, temperature=5.0)
#        choice = np.random.choice(pack, p=pick_probabilities)
#        pack.remove(choice)
#        self._update_preferences(choice, discount=convert_to_discount_factor(pick_num))
#        self.archytype_preferences_history.append(self.archytype_preferences.copy())
#        self.archytype_power_seen_history.append(self.archytype_power_seen.copy())
#        self.cards.append(choice)
#
#    def _update_power_seen(self, pack):
#        for card in pack:
#            card_values = self.parent.card_values.get(card['name'], {})
#            for arch in self.parent.deck_archetypes:
#                self.archytype_power_seen[arch] += card_values.get(arch, 0)
#
#    def _update_preferences(self, card, discount=0.5):
#        card_values = self.parent.card_values.get(card['name'], {})
#        for arch in self.parent.deck_archetypes:
#            self.archytype_preferences[arch] += discount * card_values.get(arch, 0)
#        # self._increment_open_archytype()
#
#    def _calculate_score(self, card):
#        card_values = self.parent.card_values.get(card['name'], {})
#        raw_score = sum(self.archytype_preferences[arch] * card_values.get(arch, 0)
#                    for arch in self.parent.deck_archetypes)
#        pick_number_discount = self._calcualte_pick_number_discount(card)
#        return pick_number_discount * raw_score
#
#    def _calcualte_pick_number_discount(self, card):
#        pick_number = len(self.cards)
#        if len(card['colorIdentity']) == 1:
#            return 1.0
#        elif len(card['colorIdentity']) >= 2:
#            return multi_color_sigmoid(pick_number)
#        else:
#            return zero_color_sigmoid(pick_number)
#
#    def _increment_open_archytype(self):
#        open_archytype_increments = noramlize_dict(
#            self.archytype_power_seen, scale=OPEN_ARCHYTYPE_SCALE)
#        for arch in self.parent.deck_archetypes:
#            self.archytype_preferences[arch] += open_archytype_increments[arch]
    

class Set:

    def __init__(self, cards, card_values):
        self.commons, self.uncommons, self.rares = self.split_by_rarity(cards)
        self.card_names = self.make_card_names(cards)
        self.n_cards = len(self.card_names)

    def make_card_names(self, cards):
        card_names = []
        for card in cards:
            card_names.append(card['name'])
        return card_names

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
