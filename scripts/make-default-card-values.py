import sys
import json
import itertools

colors = "WUBRG"
color_pairs = list(itertools.combinations(colors, 2))

set_json = sys.argv[1]

with open(set_json, 'rb') as f:
    m19 = json.load(f)

card_ratings = {}
for card in m19:
    color_identity = card['colorIdentity']
    card_rating = {''.join(color_pair):
                      1 if set(color_identity) & set(color_pair) else 0
                   for color_pair in color_pairs}
    card_ratings[card['name']] = card_rating

json.dump(card_ratings, sys.stdout)
