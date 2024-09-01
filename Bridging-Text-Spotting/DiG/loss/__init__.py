from __future__ import absolute_import

from .seqCrossEntropyLoss import SeqCrossEntropyLoss
from .seqLabelSmoothingCrossEntropyLoss import SeqLabelSmoothingCrossEntropyLoss
from .seqSimCLRLoss import SeqSimCLRLoss

from torch.nn.modules.loss import *
from .aiaw_loss import AIAWLoss
from .aiaw_loss_no_low_high import AIAWLoss_NO_LOW_HIGH