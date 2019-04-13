import json
import random
from math import exp
import numpy as np

M19_CARDS = json.load(open('data/m19-subset.json'))
M19_CARD_VALUES = json.load(open('data/m19-card-values.json'))
M19_DECK_ARCHYTYPES = ("WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG")


class Draft:

    def __init__(self, 
                 n_drafters=8,
                 n_rounds = 3,
                 card_values=M19_CARD_VALUES):
        self.n_drafters = n_drafters
        self.n_rounds = n_rounds
        self.drafters = [Drafter() for _ in range(n_drafters)]
        self.card_values = card_values
    
    def draft(self):
        for _ in range(self.n_rounds):
            packs = [Pack.random_pack() for _ in range(self.n_drafters)]
            self._draft_packs(packs)
        return self.drafters
    
    def _draft_packs(self, packs):
        while len(packs[0]) > 0:
            draft_pack_pairs = zip(self.drafters, packs)
            for drafter, pack in draft_pack_pairs:
                drafter.pick(pack)
            packs = rotate_list(packs)

class Drafter:

    def __init__(self, archytype_preferences=None):
        if not archytype_preferences:
            archytype_preferences = {arch: 1 for arch in M19_DECK_ARCHYTYPES}
        self.archytype_preferences = archytype_preferences
        self.cards = []

    def pick(self, pack):
        #choice = pack.pop(random.randrange(len(pack)))
        pick_scores = [self.calculate_score(card) for card in pack]
        pick_probabilities = convert_to_probabilities(pick_scores)
        choice = np.random.choice(pack, p=pick_probabilities)
        pack.remove(choice)
        self._update_preferences(choice)
        self.cards.append(choice)

    def _update_preferences(self, card):
        card_value = M19_CARD_VALUES[card['name']]
        for arch in M19_DECK_ARCHYTYPES:
            self.archytype_preferences[arch] += card_value[arch]

    def calculate_score(self, card):
        card_values = M19_CARD_VALUES[card['name']]
        score = sum(self.archytype_preferences[arch] * 2 * card_values[arch]
                    for arch in M19_DECK_ARCHYTYPES)
        return score
        
class Pack:

    @staticmethod
    def random_pack(size=12):
        pack = []
        for _ in range(size):
            pack.append(random.choice(M19_CARDS))
        return pack


def rotate_list(lst):
    first, rest = lst[0], lst[1:]
    rest.append(first)
    return rest
 
def convert_to_probabilities(scores):
    scores = np.asarray(scores)
    scores_exp = np.exp(scores)
    return scores_exp / np.sum(scores_exp) 


if __name__ == '__main__':
    draft = Draft()
    drafters = draft.draft()
    #print(list(d.cards for d in drafters))
    #print(list(d.archytype_preferences for d in drafters))
    #print()
    for i, drafter in enumerate(drafters):
        print(f"Picks for {i}'th drafter:")
        print(list(card['name'] for card in drafter.cards))
        print()
