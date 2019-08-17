import sys
import json
import itertools

colors = "WUBRG"
color_pairs = list(itertools.combinations(colors, 2))

rarity_modifiers = {
    'common': 0.0,
    'uncommon': 0.25,
    'rare': 0.5,
    'mythic': 0.5
}

set_json = sys.argv[1]
with open(set_json, 'rb') as f:
    m19 = json.load(f)


def card_weight(color_identity, color_pair, rarity):
    if set(color_identity) & set(color_pair):
        return (
            len(set(color_identity) & set(color_pair)) / len(color_identity)
            + rarity_modifiers.get(rarity, 0.0))
    else:
        return 0


card_ratings = {}
for card in m19:
    color_identity = card['colorIdentity']
    rarity = card['rarity']
    if color_identity == []:
        card_weights = {
            ''.join(color_pair): 1/3  + rarity_modifiers.get(rarity, 0.0)
            for color_pair in color_pairs}
    elif len(color_identity) >= 1:
        card_weights = {
            ''.join(color_pair): card_weight(color_identity, color_pair, rarity)
            for color_pair in color_pairs}
    card_ratings[card['name']] = card_weights

json.dump(card_ratings, sys.stdout)
