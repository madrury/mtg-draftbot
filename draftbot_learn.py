import torch


class DraftBotModel(torch.nn.Module):
    
    def __init__(self, *, n_cards, n_archetypes):
        super().__init__()
        self.n_cards = n_cards
        self.n_archetypes = n_archetypes
        self.weights = torch.nn.Parameter(
            torch.FloatTensor(n_cards, n_archetypes).uniform_(0.0, 1.0))
    
    def forward(self, X):
        options, cards = X[:, :self.n_cards], X[:, self.n_cards:]
        archetype_preferences = cards @ self.weights + 1
        option_weights = (
            options.view((X.shape[0], self.n_cards, 1))
            * self.weights.reshape((1, self.n_cards, self.n_archetypes)))
        current_option_preferences = torch.einsum(
            'pw,pcw->pc', archetype_preferences, option_weights)
        log_probs = stable_non_zero_log_softmax(current_option_preferences)
        return log_probs


def stable_non_zero_log_softmax(x):
    b = x.max(dim=1).values.view(-1, 1)
    stabalized_x = (x - b * x.sign())
    log_sum_exps = torch.log(torch.sum(x.sign() * torch.exp(stabalized_x), dim=1))
    log_probs = x.sign() * (stabalized_x - log_sum_exps.view(-1, 1))
    return log_probs
