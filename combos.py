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
            pickle.dump(combos[i*step:(i+1)*step], handle, protocol=3) # because Python 3.6 in Colab

def retrieve_combos(name) -> List[Tuple[int, int, int]]:
    with open(name, 'rb') as handle:
        return pickle.load(handle)

def sanity_check():
    from typing import List, Tuple
    combos: List[List[Tuple[int, int, int]]] = [retrieve_combos(name) for name in NAMES]
    params = parameters()
    # Do the combos have an intersection?
    for x in combos:
        for y in combos:
            if x == y:
                continue
            assert not set(x).intersection(set(y)), 'common intersection'
    # Do they add up to params?
    summed_params = [set(), set(), set()]
    for name in combos:
        for x,y,z in name:
            summed_params[0].add(params[0][x]) # batch_size
            summed_params[1].add(params[1][y]) # loss function
            summed_params[2].add(params[2][z]) # optimiser
    assert all(len(set(params_orig).intersection(params_summed)) == len(params_orig) for params_orig, params_summed in zip(params, summed_params)), "don't add up"

if __name__ == '__main__':
    seed(420) # for repeatability
    create_combos()
    sanity_check()
