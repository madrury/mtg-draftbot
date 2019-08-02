import sys
import json

j = json.load(sys.stdin)

archytype_names = j['archetype_names']
archytype_values = j['values']

card_values = {}
for name, values in archytype_values.items():
    card_values[name] = dict(zip(archytype_names, values))

json.dump(card_values, sys.stdout)
