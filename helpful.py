# LOSS FUNCTIONS
from functools import partial
from typing import Iterable, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

# Can you tell this is being monkey-patched together?
class VanillaUNET:
    DEFAULT_NUMBER_OF_CLASSES = 4
device = torch.device('cuda:0')
# anyway...

CrossEntropyLoss = nn.CrossEntropyLoss

def SmoothDiceScore(x: torch.Tensor, y: torch.Tensor, class_labels: Iterable[int]=None, smooth: float=1.0) -> torch.Tensor:
    '''
        Computes the smooth Dice score between two input tensors, with an optional
        Iterable of class labels to compute the Dice score on.

        Parameters
        ----------
        x: torch.Tensor
            Of expected dimension [batch_size, 96, 96]
        y: torch.Tensor
            Of expected dimension [batch_size, 96, 96]
        class_labels: Iterable[int], optional
            Optional class labels to solely compute the Dice score for.
        smooth: float, default 1.0
            If non-zero, computes a smooth Dice score instead.

        Notes
        -----
        This function requires both input Tensors have the same dimensions.

        Returns
        -------
        torch.Tensor
    '''
    assert x.shape == y.shape, f'bruh: {x.shape}, {y.shape}'
    assert x.dim() == y.dim() == 3, f'expected three-dimensional input, received {x.dim()}-D and {y.dim()}-D'
    batch_size: int = x.size(0)
    dice_score: torch.Tensor
    if class_labels is None:
        # Reshape Tensors
        x = x.reshape(batch_size, -1)
        y = y.reshape(batch_size, -1)
        # Compute intersection for each row
        intersection: torch.Tensor = (x == y).sum(1)
        # Compute smooth Dice score
        dice_score: torch.Tensor = (2 * intersection + smooth) / (x.size(1) + y.size(1) + smooth)
    else:
        dice_score: torch.Tensor = torch.zeros(len(class_labels))
        for i, label in enumerate(class_labels):
            # Obtain boolean indices
            x_i: torch.Tensor = x == label
            y_i: torch.Tensor = y == label
            # Compute intersection
            intersection: torch.Tensor = (x_i * y_i).sum()
            # Compute smooth Dice score
            dice_score[i] = (2 * intersection + smooth) / (x_i.sum() + y_i.sum() + smooth)
    return dice_score

class SmoothDiceLoss(nn.Module):

    def __init__(self, smooth: float=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''
            Computes the multi-class Dice loss average between U-NET's output and the correct mask.

            Parameters
            ----------
            output: torch.Tensor
                U-NET's output, dimension [batch_size, 4, 96, 96]
            mask: torch.Tensor
                Correct mask, dimension [batch_size, 96, 96]
        '''
        assert output.dim() == mask.dim() + 1, f'output.dim(): {output.dim()}, mask.dim(): {mask.dim()}'
        assert output.shape[2:] == mask.shape[1:], f'output.shape[2:]: {output.shape[2:]}, mask.shape[1:]: {mask.shape[1:]}'
        assert output.size(0) == mask.size(0), f'output.size(0): {output.size(0)}, mask.size(0): {mask.size(0)}'
        _, prediction = torch.max(output, 1)    # [batch_size, 4, 96, 96] => [batch_size, 96, 96]
        batch_size: int = output.size(0)
        dice_loss: torch.Tensor = torch.ones(batch_size, requires_grad=True).to(device) - SmoothDiceScore(prediction, mask, smooth=self.smooth).to(device)
        dice_loss = dice_loss.mean()
        return dice_loss

class FocalLoss(nn.Module):

    def __init__(self, gamma: float=2.0, alpha: float=.25):
        '''
            Initialisation for the Focal loss class.

            Parameters
            ----------
            gamma: float, default 2.0
            alpha: float, default 0.25
        '''
        super().__init__()
        self.gamma = gamma
        alphas: Iterable[float] = tuple(
            alpha for _ in range(VanillaUNET.DEFAULT_NUMBER_OF_CLASSES)
        )
        self.alpha = torch.Tensor(alphas).to(device)
        self.nll_loss = nn.NLLLoss(weight=self.alpha)

    def __str__(self) -> str:
        return f'FocalLoss(γ={self.gamma}, α={self.alpha[0]})'
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
            Computes the multi-class Focal loss between the output Tensor and
            the target Tensor
            
            Parameters
            ----------
            output: torch.Tensor
                The tensor outputted by the network
            target: torch.Tensor
                The target tensor
            
            Notes
            -----

            This implementation was only made possible thanks to this one <3
                https://github.com/AdeelH/pytorch-multi-class-focal-loss            
        '''
        if output.ndim > 2:
            c = output.shape[1]
            output = output.permute(0, *range(2, output.ndim), 1).reshape(-1, c)
            target = target.view(-1)
        # Compute weighted cross-entropy term: -alpha * log(pt)
        log_p = nn.functional.log_softmax(output, dim=-1)
        ce = self.nll_loss(log_p, target)
        # Get the True class column for each row
        all_rows = torch.arange(len(output))
        log_pt = log_p[all_rows, target]
        # Compute the Focal term: (1 - pt) ** self.gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma
        # Compute the Focal loss:
        loss = focal_term * ce
        loss = loss.mean()
        loss.requires_grad_(True)
        return loss

class CombinedLoss(nn.Module):
    
    def __init__(self, *loss_functions: Iterable[nn.Module]):
        super().__init__()
        self.loss_functions: Tuple[nn.Modules] = tuple(loss_functions)

    def __str__(self) -> str:
        return '+'.join(map(str, self.loss_functions))
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
            Applies multiple loss functions to obtain the summed loss between
            the Tensor outputted by the network and the target Tensor.

            Parameters
            ----------
            output: torch.Tensor
                Tensor outputted by the network
            target: torch.Tensor
                Target Tensor

            Returns
            -------
            torch.Tensor
                The summed loss
        '''
        return sum(LF(output, target) for LF in self.loss_functions)

# OPTIMISERS
SGD         = partial(optim.SGD, nesterov=True)
Adam        = optim.Adam
AdamW       = optim.AdamW
Adadelta    = optim.Adadelta
