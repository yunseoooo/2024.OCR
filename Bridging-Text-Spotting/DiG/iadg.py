import torch
import torch.nn as nn
from torch.nn import init

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        # init.uniform(m.weight.data, 1.0, 0.02)
        init.uniform(m.weight.data, 0.02, 1.0)
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
        self.cov_matrix_layer_low = CovMatrix_AIAW_low(dim=dim, relax_denom=0)
        self.cov_matrix_layer_high = CovMatrix_AIAW_high(dim=dim, relax_denom=0)

        init_weights(self.conv_final, init_type='kaiming')

    def forward(self, org_input, aug_input, contrast_label, cal_covstat=False, apply_wt=True, is_training=True):
        # 기존 apply_shade는 CSA layer 적용 여부와 관련되어서 뺌
        B, C, H, W = org_input.size()
        if cal_covstat:   # validate_for_cov_stat()함수가 실행될 때 True로 설정  (instance whitening을 위한 mask를 업데이트 하기 위해 ?) ㅇㅅㅇ .. 
            fea_org = self.conv_final(org_input)
            fea_aug = self.conv_final(aug_input)
            fea_org, fea_aug = org_input, aug_input

            HW = H * W

            for i in range(B):
                fea_org_sample = fea_org[i].unsqueeze(0)
                fea_aug_sample = fea_aug[i].unsqueeze(0)
                fea_org_aug_sample = torch.cat((fea_org_sample, fea_aug_sample), dim=0)
                fea_org_aug_sample = fea_org_aug_sample.view(2, C, -1)

                # 20 이하의 값이 모두인 경우 처리
                if all(label <= 20 for label in contrast_label):
                    if contrast_label[i] == min(contrast_label):  # 최소값일 때 low로 처리
                        eye, reverse_eye = self.cov_matrix_layer_low.get_eye_matrix()
                        f_covariance = torch.bmm(fea_org_aug_sample, fea_org_aug_sample.transpose(1, 2)).div(HW - 1) + (self.eps * eye)
                        off_diag_elements = f_covariance * reverse_eye
                        self.cov_matrix_layer_low.set_pair_covariance(torch.var(off_diag_elements, dim=0))
                    else:  # 나머지
                        eye, reverse_eye = self.cov_matrix_layer_high.get_eye_matrix()
                        f_covariance = torch.bmm(fea_org_aug_sample, fea_org_aug_sample.transpose(1, 2)).div(HW - 1) + (self.eps * eye)
                        off_diag_elements = f_covariance * reverse_eye
                        self.cov_matrix_layer_high.set_pair_covariance(torch.var(off_diag_elements, dim=0))
                
                # 20 이하의 값이 일부인 경우 기존 로직 유지
                elif any(label <= 20 for label in contrast_label):
                    if contrast_label[i] <= 20:
                        eye, reverse_eye = self.cov_matrix_layer_low.get_eye_matrix()
                        f_covariance = torch.bmm(fea_org_aug_sample, fea_org_aug_sample.transpose(1, 2)).div(HW - 1) + (self.eps * eye)
                        off_diag_elements = f_covariance * reverse_eye
                        self.cov_matrix_layer_low.set_pair_covariance(torch.var(off_diag_elements, dim=0))
                    else:
                        eye, reverse_eye = self.cov_matrix_layer_high.get_eye_matrix()
                        f_covariance = torch.bmm(fea_org_aug_sample, fea_org_aug_sample.transpose(1, 2)).div(HW - 1) + (self.eps * eye)
                        off_diag_elements = f_covariance * reverse_eye
                        self.cov_matrix_layer_high.set_pair_covariance(torch.var(off_diag_elements, dim=0))
                else:
                    # 20 이하가 아예 없는 경우
                    if contrast_label[i] == min(contrast_label):
                        eye, reverse_eye = self.cov_matrix_layer_low.get_eye_matrix()
                        f_covariance = torch.bmm(fea_org_aug_sample, fea_org_aug_sample.transpose(1, 2)).div(HW - 1) + (self.eps * eye)
                        off_diag_elements = f_covariance * reverse_eye
                        self.cov_matrix_layer_low.set_pair_covariance(torch.var(off_diag_elements, dim=0))
                    else:
                        eye, reverse_eye = self.cov_matrix_layer_high.get_eye_matrix()
                        f_covariance = torch.bmm(fea_org_aug_sample, fea_org_aug_sample.transpose(1, 2)).div(HW - 1) + (self.eps * eye)
                        off_diag_elements = f_covariance * reverse_eye
                        self.cov_matrix_layer_high.set_pair_covariance(torch.var(off_diag_elements, dim=0))
            return 0

        if is_training:
            fea_x2_x2 = self.conv_final(org_input)
            fea_x2_x2_aug = self.conv_final(aug_input)

            fea_org_low, fea_org_high = [], []
            fea_aug_low, fea_aug_high = [], []

            # 동일한 로직 적용
            if all(label <= 20 for label in contrast_label):
                min_label = min(contrast_label)
                for i in range(len(contrast_label)):
                    if contrast_label[i] == min_label:
                        fea_org_low.append(fea_x2_x2[i:i+1, :, :, :])
                        fea_aug_low.append(fea_x2_x2_aug[i:i+1, :, :, :])
                    else:
                        fea_org_high.append(fea_x2_x2[i:i+1, :, :, :])
                        fea_aug_high.append(fea_x2_x2_aug[i:i+1, :, :, :])
            elif any(label <= 20 for label in contrast_label):
                for i in range(len(contrast_label)):
                    if contrast_label[i] <= 20:
                        fea_org_low.append(fea_x2_x2[i:i+1, :, :, :])
                        fea_aug_low.append(fea_x2_x2_aug[i:i+1, :, :, :])
                    else:
                        fea_org_high.append(fea_x2_x2[i:i+1, :, :, :])
                        fea_aug_high.append(fea_x2_x2_aug[i:i+1, :, :, :])
            else:
                min_label = min(contrast_label)
                for i in range(len(contrast_label)):
                    if contrast_label[i] == min_label:
                        fea_org_low.append(fea_x2_x2[i:i+1, :, :, :])
                        fea_aug_low.append(fea_x2_x2_aug[i:i+1, :, :, :])
                    else:
                        fea_org_high.append(fea_x2_x2[i:i+1, :, :, :])
                        fea_aug_high.append(fea_x2_x2_aug[i:i+1, :, :, :])

            fea_org_low = torch.cat(fea_org_low, dim=0)
            fea_aug_low = torch.cat(fea_aug_low, dim=0)
            fea_org_high = torch.cat(fea_org_high, dim=0)
            fea_aug_high = torch.cat(fea_aug_high, dim=0)

            if apply_wt == True:
                eye_low, mask_matrix_low, margin_low, num_remove_cov_low = self.cov_matrix_layer_low.get_mask_matrix()
                eye_high, mask_matrix_high, margin_high, num_remove_cov_high = self.cov_matrix_layer_high.get_mask_matrix()

        else:  # test
            fea_x2_x2 = org_input

        outputs = {}
        outputs["out_feat"] = fea_x2_x2

        if is_training and apply_wt == True:
            outputs["org_feat_low"] = fea_org_low
            outputs["org_feat_high"] = fea_org_high
            outputs["aug_feat_low"] = fea_aug_low
            outputs["aug_feat_high"] = fea_aug_high
            outputs["eye_low"] = eye_low
            outputs["eye_high"] = eye_high
            outputs["mask_matrix_low"] = mask_matrix_low
            outputs["mask_matrix_high"] = mask_matrix_high
            outputs["margin_low"] = margin_low
            outputs["margin_high"] = margin_high
            outputs["num_remove_cov_low"] = num_remove_cov_low
            outputs["num_remove_cov_high"] = num_remove_cov_high

        return outputs

class CovMatrix_AIAW_low:
    def __init__(self, dim, relax_denom=0):
        super(CovMatrix_AIAW_low, self).__init__()
        self.dim = dim  # 공분산 행렬의 차원 (==input feature map channel 수)
        self.i = torch.eye(dim, dim).cuda()  # 대각성분은 1이고 나머지는 0으로 채워진 dimxdim 크기의 텐서 만들어죠 

        self.reversal_i = torch.ones(dim, dim).triu(diagonal=1).cuda() # 대각 성분은 0이고 주대각선 위 요소는 1로 채워진 상삼각 행렬 만들어죠 

        self.num_off_diagonal = torch.sum(self.reversal_i)
        self.num_sensitive = 0
        self.cov_matrix = None   # 공분산 행렬을 저장하는 변수 
        self.count_pair_cov = 0
        self.mask_matrix = None  # sensitive한 공분산 element를 마스킹하는 행렬
        print("num_off_diagonal", self.num_off_diagonal)
        if relax_denom == 0:
            print("relax_denom == 0!!!!!")
            self.margin = 0
        else:
            self.margin = self.num_off_diagonal // relax_denom 

    def get_eye_matrix(self):
        return self.i, self.reversal_i

    def reset_mask_matrix(self):
        self.mask_matrix = None


    def set_mask_matrix(self):  # 공분산 행렬에서 가장 sensitive한 top k 요소를 선택해 masking matrix를 생성하는 함수 
        # torch.set_printoptions(threshold=500000)
        self.cov_matrix = self.cov_matrix / self.count_pair_cov
        cov_flatten = torch.flatten(self.cov_matrix)  # 공분산 행렬을 flatten한 것 

        if self.margin == 0:    
            num_sensitive = int(3/1000 * cov_flatten.size()[0])  # top k개 
            print('cov_flatten.size()[0]', cov_flatten.size()[0])
            print("num_sensitive =", num_sensitive)
            _, indices = torch.topk(cov_flatten, k=int(num_sensitive))  # indices : top k개에 대한 index를 나타냄 
        else:                   # do not use
            num_sensitive = self.num_off_diagonal - self.margin
            print("num_sensitive = ", num_sensitive)
            _, indices = torch.topk(cov_flatten, k=int(num_sensitive))
        mask_matrix = torch.flatten(torch.zeros(self.dim, self.dim).cuda())
        mask_matrix[indices] = 1   # top k에 해당하는 index 위치만을 1로 갖는 mask 생성 

        if self.mask_matrix is not None:
            self.mask_matrix = (self.mask_matrix.int() & mask_matrix.view(self.dim, self.dim).int()).float()
        else:
            self.mask_matrix = mask_matrix.view(self.dim, self.dim)
        self.num_sensitive = torch.sum(self.mask_matrix)
        print("Check whether two ints are same", num_sensitive, self.num_sensitive)

        self.var_matrix = None
        self.count_var_cov = 0

        if torch.cuda.current_device() == 0:
            print("Covariance Info: (CXC Shape, Num_Off_Diagonal)", self.mask_matrix.shape, self.num_off_diagonal)
            print("Selective (Sensitive Covariance)", self.num_sensitive)

    def get_mask_matrix(self, mask=True):
        if self.mask_matrix is None:
            self.set_mask_matrix()
        return self.i, self.mask_matrix, 0, self.num_sensitive

    def set_pair_covariance(self, pair_cov):  # 공분산 행렬을 업데이트하는 함수 .. (새로운 공분산 값을 현재 공분산 행렬에 더하는 ?)
        if self.cov_matrix is None:
            self.cov_matrix = pair_cov
        else:
            self.cov_matrix = self.cov_matrix + pair_cov
        self.count_pair_cov += 1

        
class CovMatrix_AIAW_high:
    def __init__(self, dim, relax_denom=0):
        super(CovMatrix_AIAW_high, self).__init__()

        self.dim = dim
        self.i = torch.eye(dim, dim).cuda()

        # print(torch.ones(16, 16).triu(diagonal=1))
        self.reversal_i = torch.ones(dim, dim).triu(diagonal=1).cuda()

        # num_off_diagonal = ((dim * dim - dim) // 2)  # number of off-diagonal
        self.num_off_diagonal = torch.sum(self.reversal_i)
        self.num_sensitive = 0
        self.cov_matrix = None
        self.count_pair_cov = 0
        self.mask_matrix = None
        print("num_off_diagonal", self.num_off_diagonal)
        if relax_denom == 0:
            print("relax_denom == 0!!!!!")
            self.margin = 0
        else:                   # do not use
            self.margin = self.num_off_diagonal // relax_denom

    def get_eye_matrix(self):
        return self.i, self.reversal_i

    def get_mask_matrix(self, mask=True):
        if self.mask_matrix is None:
            self.set_mask_matrix()
        return self.i, self.mask_matrix, 0, self.num_sensitive

    def reset_mask_matrix(self):
        self.mask_matrix = None

    def set_mask_matrix(self):
        # torch.set_printoptions(threshold=500000)
        self.cov_matrix = self.cov_matrix / self.count_pair_cov
        cov_flatten = torch.flatten(self.cov_matrix)

        if self.margin == 0:    
            num_sensitive = int(0.6/1000 * cov_flatten.size()[0])
            print('cov_flatten.size()[0]', cov_flatten.size()[0])
            print("num_sensitive =", num_sensitive)
            _, indices = torch.topk(cov_flatten, k=int(num_sensitive))
        else:                   # do not use
            num_sensitive = self.num_off_diagonal - self.margin
            print("num_sensitive = ", num_sensitive)
            _, indices = torch.topk(cov_flatten, k=int(num_sensitive))
        mask_matrix = torch.flatten(torch.zeros(self.dim, self.dim).cuda())
        mask_matrix[indices] = 1

        if self.mask_matrix is not None:
            self.mask_matrix = (self.mask_matrix.int() & mask_matrix.view(self.dim, self.dim).int()).float()
        else:
            self.mask_matrix = mask_matrix.view(self.dim, self.dim)
        self.num_sensitive = torch.sum(self.mask_matrix)  
        print("Check whether two ints are same", num_sensitive, self.num_sensitive)

        self.var_matrix = None
        self.count_var_cov = 0

        if torch.cuda.current_device() == 0:
            print("Covariance Info: (CXC Shape, Num_Off_Diagonal)", self.mask_matrix.shape, self.num_off_diagonal)
            print("Selective (Sensitive Covariance)", self.num_sensitive)

    def set_pair_covariance(self, pair_cov):
        if self.cov_matrix is None:
            self.cov_matrix = pair_cov
        else:
            self.cov_matrix = self.cov_matrix + pair_cov
        self.count_pair_cov += 1