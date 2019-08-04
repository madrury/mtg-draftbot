# Algorithmic Magic the Gathering Drafting

This library implements an algorithmic strategy for [drafting](https://mtg.gamepedia.com/Booster_Draft) [Magic: The Gathering](https://en.wikipedia.org/wiki/Magic:_The_Gathering), in the spirit of [MTG: Arena](https://en.wikipedia.org/wiki/Magic:_The_Gathering_Arena). It supports the ability to simulate drafters, and learning a draft strategy from past draft data.

## Installation

The `mtg-draftbot` library can be installed directly from github.

```
pip install git+https://github.com/madrury/mtg-draftbot.git
```

To use some of the data processing scripts provided, you will need to install the `jq` json processing tool.

## Simulation

### Algotithm Description

The basic simulation algorithm depends on enumerating the deck archetypes in the set. If you are unfamiliar with the concept of deck archetypes, [this article](https://www.channelfireball.com/articles/ranking-the-archetypes-of-core-set-2020-draft/) discusses their definition for the most recent set (as of writing this readme).  It is a common default that deck archetypes are defined by the ten color pairs. 

Each card in the set is given a weight measuring its desirability within the given archetype:

```
{
    "Act of Treason": {
        "WU": 0.0, "WB": 0.0, "WR": 1.0, "WG": 0.0, "UB": 0.0,
	"UR": 1.0, "UG": 0.0, "BR": 1.0, "BG": 0.0, "RG": 1.0}, 
    "Aerial Assault": {
        "WU": 1.0, "WB": 1.0, "WR": 1.0, "WG": 1.0, "UB": 0.0,
	"UR": 0.0, "UG": 0.0, "BR": 0.0, "BG": 0.0, "RG": 0.0},
    "Aether Gust": {
        "WU": 1.1, "WB": 0.0, "WR": 0.0, "WG": 0.0, "UB": 1.1,
	"UR": 1.1, "UG": 1.1, "BR": 0.0, "BG": 0.0, "RG": 0.0},
    ...
}
```

We have provided utilities for generating default archetype weights, see below.

Given a single drafter considering a single pick from some number of available cards, the preference of the drafter for each card is computed as a two stage process. First, the preference of the drafter for each *archetype* is computed. These archetype preferences are baed on the cards chosen by the drafter in precious picks from the draft. Each currently held card contrubutes its archetype weight additively to the drafter's current preferences for each archetype. As a static rule, this is simply a matrix product:

```
drafter_archetype_preferences = current_cards_held @ card_archetype_weights
```

Given these archetype preferences, the drafter's preference for each available *card* is computed as a dot product between the drafter's current archetype preferences, and the weight of each available card in each archetype:

```
drafter_card_preference = dot(card_archetype_weights, drafter_archetype_perferences)
```

These preferences are now interpreted as log-probabilities. Applying a softmax function to these card preferences gives probabilities, and we can use them to make a pick by either taking the most likely card, or drawing from the resulting categorical distribution over the available cards.

Note that after choosing a card, the drafters archetype preferences change by adding the archetype weights for the chosen card to the drafters internal archetype preferences. In this way, each drafter develops a preference for a given archetype as the draft progresses, and they become more likely to choose cards in the preferred archetype(s).

### Getting Set Metadata

You will need some metadata on the set you would like to draft. This is supplied my [mtgjson](https://mtgjson.com/), and contains information like card names, rarities, and color identities. You'll want to download the complete JSON file for the set you are interested in from [here](https://mtgjson.com/downloads/sets/)

Once you have downloaded the set metadata file, the `scripts` directory contains some utilities for processing them into the form used by the simulation algorithm.

The `subset-json.jq` script will pull out only the fields in the raw set JSON that are needed for the simulation:

```
$ cat M20.json | ./scripts/subset-json.jq > m20-cards.json
```

### Constructing Draft Archetype Weights

The card weights per-archetype are stored in a `json` file which is loaded at draft time. This data is stored as a nested dictionary, with the keys in the inner dictonary archetype names.

```
{
    "Act of Treason": {
        "WU": 0.0, "WB": 0.0, "WR": 1.0, "WG": 0.0, "UB": 0.0,
	"UR": 1.0, "UG": 0.0, "BR": 1.0, "BG": 0.0, "RG": 1.0}, 
    "Aerial Assault": {
        "WU": 1.0, "WB": 1.0, "WR": 1.0, "WG": 1.0, "UB": 0.0,
	"UR": 0.0, "UG": 0.0, "BR": 0.0, "BG": 0.0, "RG": 0.0},
    "Aether Gust": {
        "WU": 1.1, "WB": 0.0, "WR": 0.0, "WG": 0.0, "UB": 1.1,
	"UR": 1.1, "UG": 1.1, "BR": 0.0, "BG": 0.0, "RG": 0.0},
    ...
}
```

To create a default archetype weights file, use the `make-default-card-values.py` script:

```
$ python scripts/make-default-card-values.py m20-cards.json > m20-default-weights.json
```

This script creates a default weight for each card in each archetype based on the card's color identity and rarity.

If you would like to edit these default weights,  the  that will convert this format to a dictionary of tuples format that is more convenient for editing:

```
$ python scripts/tuples-to-dicts.py < m20-weights-tuples.json

{
    "archetype_names": [
        "WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG"],
    "values": {
        "Act of Treason": [
	    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 
	"Aerial Assault": [
	    1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
	"Aether Gust":    [
	    1.1, 0.0, 0.0, 0.0, 1.1, 1.1, 1.1, 0.0, 0.0, 0.0],
        ...
    }
}
```

After editing to your liking, you can convert back into the nested dictionary format with :

```
$ python scripts/tuples-to-dicts.py < m20-weights-tuples.json
```

Both the identities of the deck archetypes, and the values of weights themselves can be learned from real world draft data using the machine learning features of this library, but otherwise you will have to supply them by hand. The defaults should be a good start.

### Using the Simulation Code

Once you have both your set metadata and archetype weights files prepared, you are ready to simulate some drafts.

```python
from draftbot import Draft

draft = Draft(n_drafters=8,
              n_rounds=3,
              n_cards_in_pack=14,
              cards_path='../data/m20/m20-cards.json',
              card_values_path='../data/m20/m20-default-weights.json')
draft.draft()
```

After the simulation completes, the `draft` object has a few attributes containing the history of the draft:

`draft.options: np.array, shape (n_drafters, n_cards, n_cards_in_pack * n_rounds)`

The pick options available for each drafter over each pick of the draft (i.e., what cards are left in the pack when it is passed to the drafter). Entries in this array are counts of how many of each card is available to the given drafter over each pick of the draft.

`draft.picks: np.array, shape (n_drafters, n_cards, n_cards_in_pack * n_rounds)`

Which card is chosen by each drafter over each pick of the draft. Entries in this array are either zero or one, and there is a single one in each 1-dimensional slice of the array of the form [d, :, p].

`draft.cards: np.array, shape (n_drafters, n_cards, n_cards_in_pack * n_rounds)`

The current set of cards owned by each drafter at each pick of the draft. Equal to the cumlative sum of `draft.picks` over the final axis, shifted up one index (since there are no cards owned by any player for the first
pick of the draft).

`draft.preferences: np.array, shape (n_drafters, n_archetypes, n_cards_in_pack * n_rounds)`

The preferences of each drafter for each archetype over each pick of the draft. Each 1-dimensional slice of this array of the form [d, :, p] contains the current preferences for a drafter at a single pick of the
draft.

Since these are numpy arrays, the `n_cards` axis requires some metadata to determine which cards, exactly, correspond to each position. This information is contained inside the internal `set` attribute:

```
draft.set.card_names
```

So, for example, to construct a data frame containing the draft picks for the third drafter:

```
df = pd.DataFrame(draft.picks[2, :, :], index=draft.set.card_names)
```

And to make a series contianing the names of the chosen card:

```
df.idxmax(axis=0)
```

### Plotting the Draft Picks

After the draft simulation is complete, you can easily plot the resulting draft picks.

```python
from draftbot.plotting import DraftPlotter

plotter = DraftPlotter(draft=draft)
fig, axs = plotter.plot_draft_history()
```

You'll get something like this (one copy for each drafter):

![Example of Draft Picks Plot](img/izzet-draft.png)

This plot has quite a few features:

  - Each dot represents a single draft pick. The color in the interior of the dot communicates the color identity of the card chosen, and the color of the outline represents the rarity of the card chosen.
  - The lines track the preferences of the drafter for each archetype. If the archetypes are the default color pairs, the lines will be colored accordingly.

In this example, the drafter ended up solidly in a red-blue archetype. They initially flirted with red-green, but then picked up some blue cards at the end of pack one, and took a tri-color rare (on color!) first pick in pack two.

Here's an example of a more difficult draft:

![Example of A Difficult Draft](img/izzet-draft-hard.png)

This drafter committed to red-green halfway though pack one, but had trouble picking up a good mass of green cards. Eventually the drafter settled on taking some blue cards, putting them in a bit of a three color bind. 
## Learning

The machine learning algorithm is capable of learning both the identities of deck archetypes (though not their labels) in a given set, and the weights of each card in each archetype given real world draft data.

This training data needs to be rather rich. It should contain:

  - The cards available to choose at each pick of the draft (i.e. what cards are left in the current pack).
  - The cards currently held by the drafter at each pick of the draft.
  - The chosen card at each pick of the draft.

```
Show Example Draft Data
```

Ideally, this data should be available for each drafter and each pick across many drafts. The simulation module automatically produces this data and can write it to a `sqlite` database for easy access and testing.

The machine learning library uses `pytorch` internally to train the model. You will need to create a `TensorDataset` and a `DataLoader` object to supply training batches (and another to supply test data if desired). There is no current requirement that the training batches constitute picks from a single drafter of a single draft, though this could change in the future.

To create a draftbot model and train it, use the following code outline:

```
Example of training a model
```

After training the model, the `weights` attribute contains the weights for each card in each draft archetype determined by the model:

```
...
```

## Further Work
