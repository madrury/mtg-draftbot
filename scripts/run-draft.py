import sys
from draftbot import Draft


n_drafts = int(sys.argv[1])
db_path = sys.argv[2]

for n in range(n_drafts):
    if n % 100 == 0:
        print(f"Draft {n}.")
    draft = Draft(
        cards_path='data/m20/m20-cards-subset.json',
        card_values_path='data/m20/m20-default-weights-subset.json')
    draft.draft()
    draft.write_to_database(db_path)
