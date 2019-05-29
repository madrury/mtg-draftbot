import sys
import json

card_set = json.load(open(sys.argv[1]))
card_values = json.load(open(sys.argv[2]))

card_names_to_keep = set(card_values)

reduced_set = {}
for card in card_set:
    if card['name'] in card_names_to_keep:
        reduced_set[card['name']] = card.copy()

json.dump(reduced_set, sys.stdout)
