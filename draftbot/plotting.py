import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


# This needs to be up here, since its used as a default argument to a function.
def softmax(df):
    return pd.DataFrame(
         np.exp(df).values / np.exp(df).sum(axis=1).values.reshape(-1, 1),
         columns=df.columns)


class DraftPlotter:

    def __init__(self, draft):
        self.draft = draft
        self.card_color_mapping = {
            card['name']: card['colorIdentity'] for card in draft.set.cards
        }
        self.card_rarity_mapping = {
            card['name']: card['rarity'] for card in draft.set.cards
        }

    def plot_draft_history(self, fig_width=16, single_ax_height=3):
        fig, axs = plt.subplots(
            self.draft.n_drafters,
            figsize=(fig_width, self.draft.n_drafters * single_ax_height))
        iterdata = zip(axs, self.preferences_dfs, self.cards_picked)
        for ax, preferences_df, cards in iterdata:
            self.plot_single_drafter_history(
                ax, preferences_df, cards, normalization_function=softmax)
        return fig, axs

    def plot_single_drafter_history(self,
                                    ax,
                                    history_df,
                                    cards_picked,
                                    normalization_function=softmax):
        winning_archytype, winning_history = self.plot_history_lines(
            ax,
            history_df,
            normalization_function=normalization_function)
        # Plot markers indicating the color identity and rarity of each card picked.
        # TODO: This should probably just be a speperate method.
        card_rarities = [self.card_rarity_mapping[card]
                         for card in cards_picked]
        card_colors = [self.card_color_mapping[card]
                       for card in cards_picked]
        dot_data = zip(card_colors, 
                       card_rarities, 
                       winning_history.index.values,
                       winning_history.values[:-1])
        for pick_num, (color_pair, rarity, x, y) in enumerate(dot_data):
            ax.scatter(x, y, s=170, c=[rarity_color_mapping[rarity]])
            plot_color_identity_dot(ax, x, y, color_pair)
            # TODO: This 4 should not be hard coded.
            if pick_num % 14 == 0:
                ax.axvline(pick_num, linestyle='--', alpha=0.3)
        return winning_archytype, winning_history

    def plot_history_lines(self,
                           ax, 
                           history_df, 
                           normalization_function=softmax, 
                           winning_archytype=None):

        history_df = normalization_function(history_df)
        if not winning_archytype:
            winning_archytype = history_df.iloc[-1, :].idxmax()
        winning_history = history_df.loc[:, winning_archytype]
        
        colors = make_archytype_colors(winning_archytype, alpha=1.0)
        plot_alternating_color_line(
            ax, 
            winning_history.index.values, 
            winning_history.values, 
            colors)
        
        for arch in set(history_df.columns) - set([winning_archytype]):
            arch_history = history_df.loc[:, arch]
            colors = make_archytype_colors(arch, alpha=0.2)
            plot_alternating_color_line(
                ax,
                arch_history.index.values,
                arch_history.values, colors)
        
        return winning_archytype, winning_history

    @property
    def preferences_dfs(self):
        preferences_history_dfs = []
        for drafter_idx in range(self.draft.n_drafters):
            preferences_history_dfs.append(
                pd.DataFrame(
                    self.draft.preferences[drafter_idx, :, :].T,
                    columns=self.draft.archetype_names))
        return preferences_history_dfs

    @property
    def cards_picked(self):
        cards_picked = []
        for i in range(self.draft.n_drafters):
            df = pd.DataFrame(self.draft.picks[i, :, :].astype(int).T,
                              columns=self.draft.card_names)
            cards_picked.append(df.idxmax(axis=1))
        return cards_picked


def plot_alternating_color_line(ax, x, y, colors=None):
    x, y = (
        insert_middpoints(x.astype(float)),
        insert_middpoints(y.astype(float)))
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_boundaries = insert_middpoints(np.arange(x.min(), x.max() + 1))
    n_bins = len(segment_boundaries) - 1
    cmap = ListedColormap((colors * n_bins)[:n_bins])
    norm = BoundaryNorm(segment_boundaries, cmap.N)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(x)
    lc.set_linewidth(2)
    ax.add_collection(lc)


def insert_middpoints(arr):
    midpoints = (arr[:-1] + arr[1:]) / 2
    return np.insert(arr, np.arange(1, len(arr)), midpoints)


def plot_color_identity_dot(ax, x, y, color_identity, s=100, alpha=1.0):
    arch_color_mapping = make_archytype_color_mapping(alpha)
    colors = [arch_color_mapping[c] for c in color_identity]
    if len(colors) == 0:
        ax.scatter(x, y, c=['grey'], s=s, zorder=2)
    if len(colors) == 1:
        ax.scatter(x, y, c=[colors[0]], s=s, zorder=2)
    if len(colors) >= 2:
        plot_multi_color_dot(ax, x, y, colors)


def plot_multi_color_dot(ax, x, y, colors, s=100):
    angles = list(np.linspace(start=0, stop=2*np.pi, num=(len(colors) + 1)))
    begin_angles = angles[:-1]
    end_angles = angles[1:]
    for angle, end_angle, color in zip(begin_angles, end_angles, colors):
        xs = [0] + np.sin(np.linspace(angle, end_angle, 25)).tolist()
        ys = [0] + np.cos(np.linspace(angle, end_angle, 25)).tolist()
        xy = np.column_stack([xs, ys])
        ax.scatter(x, y, marker=xy, s=s, facecolor=color, zorder=2)


def make_archytype_color_mapping(alpha):
    """Constuct a dictionary mapping MTG color identities to physical RGBA
    colors.
    """
    arch_color_mapping = {
        'W': [1, 0.8, 0.5, alpha],
        'U': [0.2, 0.2, 1, alpha],
        'B': [0.2, 0.2, 0.2, alpha],
        'R': [1, 0, 0, alpha],
        'G': [0, 0.8, 0, alpha]
    }
    return arch_color_mapping


def make_archytype_colors(arch, alpha=1):
    """Return the RGBA tuples for a given archetype defined by some number of
    MTG colors.
    """
    arch_color_mapping = make_archytype_color_mapping(alpha)
    return [arch_color_mapping[c] for c in arch]


rarity_color_mapping = {
    'common': [0.0, 0.0, 0.0],
    'uncommon': [192 / 255, 192 / 255, 192 / 255],
    'rare': [255 / 255, 215 / 255, 0],
    'mythic': [255 / 255, 140 / 255, 0],    
}
