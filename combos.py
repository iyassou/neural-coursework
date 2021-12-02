from functools import partial
from itertools import chain, combinations, product
from typing import Any, List, Tuple
import helpful
from random import seed, shuffle
import pickle

NAMES = 'AlexT AlexW Amy Liv Sam'.split()

def parameters() -> List[List[Any]]:
    '''
        Returns the nested, ORDERED list of parameters.
    '''
    # BATCH_SIZES - Batch size during training.
    BATCH_SIZES = 1, 2, 5, 10, 20
    # LOSS_FUNCTIONS - Loss functions to use during training.
    # Validation loss will always be SmoothDiceLoss.
    ce_loss = helpful.CrossEntropyLoss()
    smd_loss = helpful.SmoothDiceLoss()
    focal_loss = helpful.FocalLoss()
    losses = ce_loss, smd_loss, focal_loss
    LOSS_FUNCTIONS = []
    LOSS_FUNCTIONS.extend(losses)
    LOSS_FUNCTIONS.extend(
        helpful.CombinedLoss(*combo) for combo in chain(
            combinations(losses, 2), combinations(losses, 3)
        )
    )
    # OPTIMISERS - Optimiser to use during training.
    OPTIMISERS = []
    sgd_momentums = .9, .925, .95, .975, .99
    sgd_learning_rates = .001, .005, .01, .05, .1
    OPTIMISERS.extend(
        partial(helpful.SGD, lr=learning_rate, momentum=mo)
        for mo, learning_rate in product(sgd_momentums, sgd_learning_rates)
    )
    adams_learning_rates = .01, .001
    adams_betas = (0.9, 0.99), (0.9, 0.999)
    OPTIMISERS.extend(
        partial(opt, lr=learning_rate, betas=beta)
        for beta, learning_rate in product(adams_betas, adams_learning_rates)
        for opt in (helpful.Adam, helpful.AdamW)
    )
    adadelta_rhos = .9, .95
    OPTIMISERS.extend(
        partial(helpful.Adadelta, rho=ro) for ro in adadelta_rhos
    )
    return [BATCH_SIZES, LOSS_FUNCTIONS, OPTIMISERS]

def create_combos():
    params = parameters()
    combos = list(
        product(*[list(range(len(x))) for x in params])
    )
    shuffle(combos)
    step = len(combos) // len(NAMES)
    for i, name in enumerate(NAMES):
        with open(name, 'wb') as handle:
            pickle.dump(combos[i:i+step], handle, protocol=pickle.HIGHEST_PROTOCOL)

def retrieve_combos(name) -> List[Tuple[int, int, int]]:
    with open(name, 'rb') as handle:
        return pickle.load(handle)

if __name__ == '__main__':
    seed(420) # for repeatability
    create_combos()
