from typing import Any, Iterable, Tuple
from functools import partial
import torch.optim as optim

# LOSS FUNCTIONS
class MockNN:
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

class SmoothDiceLoss(MockNN):
    pass

class CrossEntropyLoss(MockNN):
    pass

class FocalLoss:
    def __repr__(self) -> str:
        return f'FocalLoss(γ=2, α=0.25)'

class CombinedLoss:
    def __init__(self, *loss_functions: Iterable[Any]):
        self.loss_functions = tuple(loss_functions)
    def __repr__(self):
        return '+'.join(map(str, self.loss_functions))

# OPTIMISERS
SGD         = partial(optim.SGD, nesterov=True)
Adam        = optim.Adam
AdamW       = optim.AdamW
Adadelta    = optim.Adadelta
