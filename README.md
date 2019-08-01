# Algorithmic Magic the Gathering Drafting

This library implements an algorithmic strategy for [drafting](https://mtg.gamepedia.com/Booster_Draft) [Magic: The Gathering](https://en.wikipedia.org/wiki/Magic:_The_Gathering), in the spirit of [MTG: Arena](https://en.wikipedia.org/wiki/Magic:_The_Gathering_Arena). It supports the ability to simulate drafters, and learning a draft strategy from past draft data.

## Installation

The `mtg-draftbot` library can be installed directly from github.

```
pip install git+https://github.com/madrury/mtg-draftbot.git
```

## Setup and Usage

### Simulation

To simulate a draft, you will need to provide the algorithm with two types of metadata.

#### Set Metadata

You will need some metadata on the set you would like to draft. This is supplied my [mtgjson](https://mtgjson.com/), and contains information like card names, rarities, and color identities. You'll want to download the complete JSON file for the set you are interested in from [here](https://mtgjson.com/downloads/sets/)

Once you have downloaded the set metadata file, the `scripts` directory contains some utilities for processing them into the form used by the simulation algorithm.

```
TODO: How to process the metadata files.  Do it for M20.
```

#### Draft Archetype Weights

The basic simulation algorithm depends on enumerating the deck archetypes in the set. If you are unfamiliar with the concept of deck archetypes, [this article](https://www.channelfireball.com/articles/ranking-the-archetypes-of-core-set-2020-draft/) discusses their definition for the most recent set (as of writing this readme).  It is a common default that deck archetypes are defined by the ten color pairs. 

Once the deck archetypes for the set are defined, each card should be given a weight in each archetype, with larger weights indicating that the card is more desirable in the given archetype, and a zero indicating that the card is not at all desired in the given archetype (i.e. a red card in a black-blue deck).

The card weights per-archetype are stored in a `json` file which is loaded at draft time.

```
TODO: Example of card weights json
```

Both the identities of the deck archetypes, and the values of weights themselves can be learned from real world draft data using the machine learning part of this library, but otherwise you will have to supply them by hand.  We have supplied some examples for the Core Set 2020 in this repository to get you started.

#### Using the Simulation Code

...


#### Plotting the Draft Picks

...

### Learning

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

## Algorithm Descriptions

As mentioned above, the draft simulation algorithm depends on a prior enumeration of the various deck archetypes in the draft format, and an assignment of weight to each card in each of these enumerated archetype. 

Given a single drafter considering a single pick from some number of available cards, the preference of the drafter for each card is computed as a two stage process. First, the preference of the drafter for each *archetype* is computed by accumulating the weights for each card currently held by the drafter. This takes the form of a simple matrix product:

```
drafter_archetype_preferences = current_cards_held @ card_archetype_weights
```

Given these archetype preferences, the drafter's preference for each available *card* is computed as a dot product between the drafter's current archetype preferences, and the weights of each available card in each archetype:

```
drafter_card_preference = dot(card_archetype_weights, drafter_archetype_perferences)
```

These preferences are now interpreted as log-probabilities. Applying a softmax function to these card preferences gives probabilities, and we can use them to make a pick by either taking the most likely card, or drawing from the resulting categorical distribution over the available cards.

Note that after choosing a card, the drafters archetype preferences change by adding the archetype weights for the chosen card to the drafters internal archetype preferences. In this way, each drafter develops a preference for a given archetype as the draft progresses, and they become more likely to choose cards in the preferred archetype(s).

## Further Work
