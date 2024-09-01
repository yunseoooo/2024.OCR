import torch
import torch.nn as nn

class AIAWLoss_NO_LOW_HIGH(nn.Module):
    def __init__(self, reduction=None):
        super(AIAWLoss_NO_LOW_HIGH, self).__init__()
        self.reduction = reduction

    def forward(self, f_map, eye, mask_matrix, margin, num_remove_cov): 
        '''
         Input:
            f_map (Tensor): The feature map, with shape (B, C, H, W), where B is the batch size, C is the number of channels, and H and W are the height and width of the feature map.
            eye (Tensor): The identity matrix used for covariance calculation.  torch.Size([128, 384, 16, 16]) torch.Size([1536, 384])
            mask_matrix (Tensor): The mask matrix used for selective covariance calculation.  torch.Size([1536, 384])
            margin (Tensor): Margin for the covariance.
            num_remove_cov (Tensor): The number of covariances to be removed.
         Return:
             loss (Tensor): The AIAW loss
         '''
        B, C, H, W = f_map.size()
        if mask_matrix.size(0) > C:
            num_gpus = mask_matrix.size(0) // C
            mask_matrix = mask_matrix.view(num_gpus, C, C).mean(dim=0)


        f_cov, B = get_covariance_matrix(f_map, eye)
        f_cov_masked = f_cov * mask_matrix

        off_diag_sum = torch.sum(torch.abs(f_cov_masked), dim=(1,2), keepdim=True) - margin  # B X 1 X 1
        loss = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0)  # B X 1 X 1
        loss = torch.sum(loss) / B

        return loss


def get_covariance_matrix(f_map, eye=None):
    '''
     Input:
        f_map (Tensor): The feature map tensor with shape (B, C, H, W), where B is the batch size, C is the number of channels, and H and W are the height and width of the feature map.
        eye (Tensor, optional): The identity matrix used for covariance calculation. Defaults to None.
     Return:
         f_cor (Tensor): The covariance matrix of the feature map
         B (int): batch size
     '''
    eps = 1e-5
    B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
    HW = H * W
    if eye is None or eye.shape[0] != C or eye.shape[1] != C:
        eye = torch.eye(C).cuda()
    f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW-1) + (eps * eye)  # C X C / HW

    return f_cor, B
