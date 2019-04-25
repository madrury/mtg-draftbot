import json
import random
from math import exp
import numpy as np


M19_CARD_VALUES = json.load(open('data/m19-custom-card-values.json'))
M19_DECK_ARCHYTYPES = ("WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "GR")

M19_CARDS = [card for card in json.load(open('data/m19-subset.json'))
                  if card['name'] in M19_CARD_VALUES]
M19_COMMONS = [card for card in M19_CARDS if card['rarity'] == 'common']
M19_UNCOMMONS = [card for card in M19_CARDS if card['rarity'] == 'uncommon']
M19_RARES = [card for card in M19_CARDS if card['rarity'] in {'rare', 'mythic'}]


class Draft:

    def __init__(self, n_drafters=8, n_rounds=3):
        self.n_drafters = n_drafters
        self.n_rounds = n_rounds
        self.drafters = [Drafter() for _ in range(n_drafters)]
        self._round = 0
        # You may not need to track this since its always equal to the number
        # of cards that a drafter has.
        self._pick_num = 0
    
    def draft(self):
        for _ in range(self.n_rounds):
            packs = [Pack.random_pack() for _ in range(self.n_drafters)]
            self._draft_packs(packs)
        return self.drafters
    
    def _draft_packs(self, packs):
        while len(packs[0]) > 0:
            for drafter, pack in zip(self.drafters, packs):
                drafter.pick(pack, self._pick_num)
            packs = rotate_list(packs, round=self._round)
            self._pick_num += 1
        self._round += 1


class Drafter:

    def __init__(self, archytype_preferences=None):
        if not archytype_preferences:
            archytype_preferences = {arch: 1 for arch in M19_DECK_ARCHYTYPES}
        self.archytype_preferences = archytype_preferences
        self.archytype_power_seen = {arch: 0 for arch in M19_DECK_ARCHYTYPES}
        self.cards = []
        self.archytype_preferences_history = [archytype_preferences.copy()]
        self.archytype_power_seen_history = []

    def pick(self, pack, pick_num):
        self._update_power_seen(pack)
        pick_scores = [self._calculate_score(card) for card in pack]
        pick_probabilities = convert_to_probabilities_softmax(pick_scores, temperature=2.0)
        choice = np.random.choice(pack, p=pick_probabilities)
        pack.remove(choice)
        self._update_preferences(choice, discount=convert_to_discount_factor(pick_num))
        self.archytype_preferences_history.append(self.archytype_preferences.copy())
        self.archytype_power_seen_history.append(self.archytype_power_seen.copy())
        self.cards.append(choice)

    def _update_power_seen(self, pack):
        for card in pack:
            card_values = M19_CARD_VALUES.get(card['name'], {})
            for arch in M19_DECK_ARCHYTYPES:
                self.archytype_power_seen[arch] += card_values.get(arch, 0)

    def _update_preferences(self, card, discount=1.0):
        card_values = M19_CARD_VALUES.get(card['name'], {})
        for arch in M19_DECK_ARCHYTYPES:
            self.archytype_preferences[arch] += discount * card_values.get(arch, 0)
        self._increment_maximum_preference()
        self._increment_open_archytype()

    def _calculate_score(self, card):
        card_values = M19_CARD_VALUES.get(card['name'], {})
        raw_score = sum(self.archytype_preferences[arch] * card_values.get(arch, 0)
                    for arch in M19_DECK_ARCHYTYPES)
        pick_number_discount = self._calcualte_pick_number_discount(card)
        return pick_number_discount * raw_score

    def _calcualte_pick_number_discount(self, card):
        pick_number = len(self.cards)
        if len(card['colorIdentity']) == 1:
            return 1.0
        elif len(card['colorIdentity']) >= 2:
            return multi_color_sigmoid(pick_number)
        else:
            return zero_color_sigmoid(pick_number)

    def _increment_maximum_preference(self):
        max_arcytype = max(self.archytype_preferences, key=self.archytype_preferences.get)
        self.archytype_preferences[max_arcytype] += 0.5
    
    def _increment_open_archytype(self):
        open_archytype = max(self.archytype_power_seen, key=self.archytype_power_seen.get)
        self.archytype_preferences[open_archytype] += 1.0

class Pack:

    @staticmethod
    def random_pack(size=14):
        n_rares, n_uncommons, n_commons = 1, 3, size - 4
        pack = []
        for _ in range(n_commons):
            pack.append(random.choice(M19_COMMONS))
        for _ in range(n_uncommons):
            pack.append(random.choice(M19_UNCOMMONS))
        pack.append(random.choice(M19_RARES))
        return pack


def rotate_list(lst, round):
    if round % 2 == 0:
        rest, last = lst[:-1], lst[-1]
        return [last] + rest
    else:
        first, rest = lst[0], lst[1:]
        rest.append(first)
        return rest
 
def convert_to_probabilities_softmax(scores, temperature=0.5):
    scores = np.asarray(scores)
    scores_exp = np.exp(scores / temperature)
    return scores_exp / np.sum(scores_exp) 

def convert_to_discount_factor(pick_num, pack_size=14):
    return 1 / (1 + np.exp((pick_num % pack_size - 10)))

def make_sigmoid(*, a=2, b=1, t0=0, rate=1):
    def sigmoid(t):
        return (b - a) / (1 + np.exp(- rate * (t - t0))) + a
    return sigmoid

zero_color_sigmoid = make_sigmoid(a=4/10, b=1, t0=3, rate=2) 
multi_color_sigmoid = make_sigmoid(a=7/10, b=1, t0=3, rate=2)