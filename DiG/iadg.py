import torch
import torch.nn as nn
from torch.nn import init

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    elif init_type == 'orthogonal_rnn':
        net.apply(weights_init_orthogonal_rnn)
    elif init_type == 'const':
        net.apply(weights_init_const)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)    

class Shader(nn.Module):
    def __init__(self, dim=384, eps=1e-7, model_initial='kaiming'):
        super(Shader, self).__init__()
        self.eps = eps
        self.conv_final = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.cov_matrix_layer = CovMatrix_AIAW(dim=dim, relax_denom=0)

        init_weights(self.conv_final, init_type='kaiming')

    def forward(self, org_input, aug_input, contrast_label, cal_covstat=False, apply_wt=True, is_training=True):
        B, C, H, W = org_input.size()

        device = org_input.device
        org_input = org_input.to(device)
        aug_input = aug_input.to(device)

        if cal_covstat:
            fea_org = self.conv_final(org_input)
            fea_aug = self.conv_final(aug_input)
            fea_org, fea_aug = org_input, aug_input

            HW = H * W

            for i in range(B):
                fea_org_sample = fea_org[i].unsqueeze(0)
                fea_aug_sample = fea_aug[i].unsqueeze(0)
                fea_org_aug_sample = torch.cat((fea_org_sample, fea_aug_sample), dim=0)
                fea_org_aug_sample = fea_org_aug_sample.view(2, C, -1).to(device)  
                
                eye, reverse_eye = self.cov_matrix_layer.get_eye_matrix()
                eye, reverse_eye = eye.to(device), reverse_eye.to(device)
                f_covariance = torch.bmm(fea_org_aug_sample, fea_org_aug_sample.transpose(1, 2)).div(HW - 1) + (self.eps * eye)
                off_diag_elements = f_covariance * reverse_eye
                self.cov_matrix_layer.set_pair_covariance(torch.var(off_diag_elements, dim=0))

            # return torch.tensor(0.0).to(device)
            return torch.tensor(0.0).to(device)

        if is_training:
            fea_x2_x2 = self.conv_final(org_input)
            fea_x2_x2_aug = self.conv_final(aug_input)

            if apply_wt:
                eye, mask_matrix, margin, num_remove_cov = self.cov_matrix_layer.get_mask_matrix()
                if mask_matrix.shape != torch.Size([384, 384]):
                    breakpoint()

        else:  # test
            fea_x2_x2 = org_input

        outputs = {}
        outputs["out_feat"] = fea_x2_x2

        if is_training and apply_wt:
            outputs["eye"] = eye
            outputs["mask_matrix"] = mask_matrix
            outputs["margin"] = margin
            outputs["num_remove_cov"] = num_remove_cov
        return outputs

class CovMatrix_AIAW:
    def __init__(self, dim, relax_denom=0):
        super(CovMatrix_AIAW, self).__init__()
        self.dim = dim
        self.i = torch.eye(dim, dim).cuda()
        self.reversal_i = torch.ones(dim, dim).triu(diagonal=1).cuda()
        self.num_off_diagonal = torch.sum(self.reversal_i)
        self.num_sensitive = 0
        self.cov_matrix = None
        self.count_pair_cov = 0
        self.mask_matrix = None
        if relax_denom == 0:
            self.margin = 0
        else:
            self.margin = self.num_off_diagonal // relax_denom 

    def get_eye_matrix(self):
        return self.i, self.reversal_i

    def reset_mask_matrix(self):
        self.mask_matrix = None

    def set_mask_matrix(self):
        self.cov_matrix = self.cov_matrix / self.count_pair_cov
        cov_flatten = torch.flatten(self.cov_matrix)

        if self.margin == 0:    
            num_sensitive = int(3/1000 * cov_flatten.size()[0])
            # num_sensitive = int(1/1000 * cov_flatten.size()[0])
            _, indices = torch.topk(cov_flatten, k=int(num_sensitive))
        else:
            num_sensitive = self.num_off_diagonal - self.margin
            _, indices = torch.topk(cov_flatten, k=int(num_sensitive))
        mask_matrix = torch.flatten(torch.zeros(self.dim, self.dim).cuda())
        mask_matrix[indices] = 1

        if self.mask_matrix is not None:
            self.mask_matrix = (self.mask_matrix.int() & mask_matrix.view(self.dim, self.dim).int()).float()
        else:
            self.mask_matrix = mask_matrix.view(self.dim, self.dim)
        self.num_sensitive = torch.sum(self.mask_matrix)

    def get_mask_matrix(self, mask=True):
        if self.mask_matrix is None:
            self.set_mask_matrix()
        return self.i, self.mask_matrix, torch.tensor(0).to(self.i.device), self.num_sensitive

    def set_pair_covariance(self, pair_cov):
        if self.cov_matrix is None:
            self.cov_matrix = pair_cov
        else:
            pair_cov = pair_cov.to(self.cov_matrix.device)
            self.cov_matrix = self.cov_matrix + pair_cov
        self.count_pair_cov += 1
