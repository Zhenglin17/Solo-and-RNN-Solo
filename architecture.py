import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import torch
from torch import nn
from torch.autograd import Variable

class GRU_cell(nn.Module):
    def __init__(self, in_channels, hidden_channels, gate_kernel):
        super(GRU_cell,self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.Wz = nn.Conv2d(in_channels, hidden_channels, padding='same', bias = True, kernel_size = gate_kernel)
        self.Uz = nn.Conv2d(hidden_channels, hidden_channels, padding='same', bias = False, kernel_size = gate_kernel)

        self.Wr = nn.Conv2d(in_channels, hidden_channels, padding='same', bias = True, kernel_size = gate_kernel)
        self.Ur = nn.Conv2d(hidden_channels, hidden_channels, padding='same', bias = False, kernel_size = gate_kernel)

        self.Wh = nn.Conv2d(in_channels, hidden_channels, padding='same', bias = True, kernel_size = gate_kernel)
        self.Uh = nn.Conv2d(hidden_channels, hidden_channels, padding='same', bias = False, kernel_size = gate_kernel)

        self.sigmoid = nn.Sigmoid()
        # leaky relu to avoid vanishing gradient
        self.leaky_relu = nn.LeakyReLU()


    def forward(self,images, hidden_state):
        # N * C * W * H
            

        zt = self.sigmoid(self.Wz(images) + self.Uz(hidden_state))
        rt = self.sigmoid(self.Wr(images) + self.Ur(hidden_state))

        h_hat = self.leaky_relu(self.Wh(images) + self.Uh(rt * hidden_state))

        return (1 - zt) * h_hat + zt * hidden_state

class GRU(nn.Module):
    def __init__(self, in_channels, hidden_channels, gate_kernel):
        super(GRU,self).__init__()

        self.cell = GRU_cell(in_channels, hidden_channels, gate_kernel)
        self.hidden_channels = hidden_channels

    def forward(self, sequence, hidden_state = None):

        # N * T * C * W * H

        if hidden_state == None:
            hidden_state = torch.zeros(sequence.shape[0], self.hidden_channels, sequence.shape[-2], sequence.shape[-1]).to(sequence.device)

        for i in range(sequence.shape[1]):
            hidden_state = self.cell(sequence[:,i],hidden_state)

        return hidden_state

class bi_GRU(nn.Module):
    def __init__(self, in_channels, hidden_channels, gate_kernel = 3):
        super(bi_GRU,self).__init__()

        self.gru = GRU(in_channels, hidden_channels, gate_kernel)
        self.hidden_channels = hidden_channels
        self.bn = nn.BatchNorm2d(hidden_channels)

        # 1 * 1 convolution to combine information
        self.l = nn.Conv2d(2 * self.hidden_channels, hidden_channels, padding='same', bias = False, kernel_size = 1)

    def forward(self, sequences):

        # N * 2 * T * C * W * H
        lags = sequences.shape[2]


        forward = self.gru(sequences[:,0])
        bacward = self.gru(torch.flip(sequences[:,1], dims = [1]))


        hidden_state = torch.cat([forward, bacward],1)

        return self.bn(self.l(hidden_state))


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
class FPN(nn.Module):
    def __init__(self, block, layers, input_channel = 1):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True) + y
    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        #print(f'c1:{c1.shape}')
        c2 = self.layer1(c1)
        #print(f'c2:{c2.shape}')
        c3 = self.layer2(c2)
        #print(f'c3:{c3.shape}')
        c4 = self.layer3(c3)
        #print(f'c4:{c4.shape}')
        c5 = self.layer4(c4)
        #print(f'c5:{c5.shape}')
        # Top-down
        p5 = self.toplayer(c5)
        #print(f'p5:{p5.shape}')
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        #print(f'latlayer1(c4):{self.latlayer1(c4).shape}, p4:{p4.shape}')
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        #print(f'latlayer1(c3):{self.latlayer2(c3).shape}, p3:{p3.shape}')
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        #print(f'latlayer1(c2):{self.latlayer3(c2).shape}, p2:{p2.shape}')
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5
    def sequence_output(self,sequence):
        # sequence: (batchsize, direction, time, channels, height, width)

        direction = sequence.shape[1]
        batch = sequence.shape[0]
        out = [[],[],[],[]]

        for i in range(batch):
            forward = self(sequence[i,0])
            backward = self(sequence[i,1])

            for j,(f,b) in enumerate(zip(forward, backward)):
                out[j].append(torch.stack([f, b]))

        
        return [torch.stack(o) for o in out]


            
class SOLO_head(nn.Module):
    def __init__(self,
                 scale_ranges,
                 num_grids, 
                 mask_loss_cfg,
                 cate_loss_cfg,
                 postprocess_cfg):  
        super(SOLO_head, self).__init__()
        self.scale_ranges = scale_ranges
        print(scale_ranges)
        self.num_grids = num_grids
        print(num_grids)
        self.mask_loss_cfg = mask_loss_cfg
        self.cate_loss_cfg = cate_loss_cfg
        self.postprocess_cfg = postprocess_cfg
        
        self.bce_loss = nn.BCELoss()

        self.cate_branch = []
        for i in range(7):
            self.cate_branch.append(nn.Conv2d(256, 256, 3, 1, 1, bias=False))
            self.cate_branch.append(nn.GroupNorm(32, 256))
            self.cate_branch.append(nn.ReLU())
        self.cate_branch.append(nn.Conv2d(256, 1, 3, 1, 1, bias=True))
        self.cate_branch.append(nn.Sigmoid())
        self.cate_branch = nn.Sequential(*self.cate_branch)
        
        self.ins_branch = []
        self.ins_branch.append(nn.Conv2d(258, 256, 3, 1, 1, bias=False))
        self.ins_branch.append(nn.GroupNorm(32, 256))
        self.ins_branch.append(nn.ReLU())
        for j in range(6):
            self.ins_branch.append(nn.Conv2d(256, 256, 3, 1, 1, bias=False))
            self.ins_branch.append(nn.GroupNorm(32, 256))
            self.ins_branch.append(nn.ReLU())
        self.ins_branch = nn.Sequential(*self.ins_branch)
        
        self.ins_outs = nn.ModuleList()
        for i in range(len(self.num_grids)):
            self.ins_outs.append(nn.Sequential(nn.Conv2d(256, self.num_grids[i]**2, 1, padding=0, bias=True), nn.Sigmoid()))


    
    def generate_single_target(self, single_bboxes, single_labels, single_mask, featmap_sizes, eps, device='cpu'):

        active_masks = []
        cate_masks = []
        target_masks = []

        center_x1 = ((1 + eps) / 2 * single_bboxes[:, 0] + (1 - eps) / 2 * single_bboxes[:, 2])
        center_x2 = ((1 - eps) / 2 * single_bboxes[:, 0] + (1 + eps) / 2 * single_bboxes[:, 2])

        center_y1 = ((1 + eps) / 2 * single_bboxes[:, 1] + (1 - eps) / 2 * single_bboxes[:, 3])
        center_y2 = ((1 - eps) / 2 * single_bboxes[:, 1] + (1 + eps) / 2 * single_bboxes[:, 3])
        target_scales = torch.sqrt((single_bboxes[:, 2] - single_bboxes[:, 0]) * (single_bboxes[:, 3] - single_bboxes[:, 1]))
        
        for fp_size, num_grid, fpn_scale in zip(featmap_sizes, self.num_grids, self.scale_ranges):
            new_x1 = center_x1 * num_grid
            new_x2 = center_x2 * num_grid
            new_y1 = center_y1 * num_grid
            new_y2 = center_y2 * num_grid
            
            cate_mask = torch.zeros(num_grid, num_grid)
            target_mask = torch.zeros(num_grid ** 2, fp_size[0]*2,fp_size[0]*2)

            for obj_idx in range(len(single_labels)):
                
                # Check scale range
                if fpn_scale[0] <= target_scales[obj_idx].item() < fpn_scale[1]:
                    
                    top, bottom, left, right = new_y1[obj_idx].int().item(), new_y2[obj_idx].int().item(), new_x1[obj_idx].int().item(), new_x2[obj_idx].int().item()
                    
                    if top >= bottom:
                        top = int(((top + bottom) / 2))
                        bottom = top + 1
                    if left >= right:
                        left = int(((left + right) / 2))
                        right = left + 1
                    cate_mask[top:bottom, left:right] = single_labels[obj_idx]
                    temp_mask = single_mask[obj_idx][None, None, :]

                    interp_mask = nn.functional.interpolate(temp_mask, size=[2 * f for f in fp_size], mode='bilinear', align_corners = True)[0,0]
                    interp_mask = (interp_mask > 0.5).int().clone()

                    pos = torch.arange(top,bottom).reshape(-1, 1) * num_grid + torch.arange(left, right)
                    pos = pos.flatten().int()
                    for p in pos:
                        target_mask[p] = interp_mask

            active_mask = (cate_mask > 0)
            active_masks.append(active_mask.bool().flatten().to(device))
            cate_masks.append(cate_mask.int().flatten().to(device))
            target_masks.append(target_mask.to(device))
    
        return target_masks, cate_masks, active_masks

    def generate_targets(self, gt_masks, gt_bboxes, gt_labels, eps, featmap_sizes=None, device='cpu'):
        if featmap_sizes is None:
            featmap_sizes = self.featmap_sizes
        cate_targets = []
        mask_targets = []
        active_masks = []

        for bbox, lab, mask in zip(gt_bboxes, gt_labels, gt_masks):
            per_img_result = self.generate_single_target(bbox, lab, mask, featmap_sizes=featmap_sizes, eps=eps, device=device)
            mask_targets.append(per_img_result[0])
            cate_targets.append(per_img_result[1])
            active_masks.append(per_img_result[2])
        return mask_targets, cate_targets, active_masks
        
    def loss(self,
             category_predictions,
             mask_predictions,
             mask_targets,
             cate_targets,
             active_masks,
             device='cpu'):
        
        mask_trues, mask_preds, cate_trues, cate_preds = [], [], [], []
        
        for target_level, active_level in zip(zip(*mask_targets), zip(*active_masks)):
            tmp = []
            for target_image, active_image in zip(target_level, active_level):
                tmp.append(target_image[active_image, :, :])
            mask_trues.append(torch.cat(tmp, axis=0))

        for pred_level, active_level in zip(mask_predictions, zip(*active_masks)):
            tmp = []
            for pred_image, active_image in zip(pred_level, active_level):
                tmp.append(pred_image[active_image, :, :])
            mask_preds.append(torch.cat(tmp, axis=0))
        
        for level in zip(*cate_targets):
            for image in level:
                cate_trues += image.flatten().tolist()
        cate_trues = torch.Tensor(cate_trues).to(device)

        for level in category_predictions:
            cate_preds.append(level.flatten())
        cate_preds = torch.cat(cate_preds).to(device)

        
        cate_loss = self.bce_loss(cate_preds,cate_trues)
#         cate_loss = self.get_cate_loss(cate_trues, cate_preds, device=device)
        mask_loss = self.get_mask_loss(mask_trues, mask_preds, device=device)
    
        
        lambda_cate, lambda_mask = self.cate_loss_cfg['weight'], self.mask_loss_cfg['weight']
        total_loss = lambda_cate * cate_loss + lambda_mask * mask_loss
        
        return cate_loss, mask_loss, total_loss
    
    def dice_loss(self, y_true, y_pred, smooth=1e-6, device='cpu'):
        y_true, y_pred = y_true.to(device), y_pred.to(device)
        intersection = 2 * torch.sum(y_true * y_pred)
        sum_true = torch.sum(y_true * y_true)
        sum_pred = torch.sum(y_pred * y_pred)
        dice_coeff = (intersection + smooth) / (sum_true + sum_pred + smooth)
        return 1. - dice_coeff
        
    def get_mask_loss(self, mask_trues, mask_preds, device='cpu'):
        '''Loss function of mask'''
        num_pos = sum([level.shape[0] for level in mask_preds])  # the number of positive samples
        summation = sum([sum([self.dice_loss(true, pred) 
                            for true, pred in zip(true_level, pred_level)])
                            for true_level, pred_level in zip(mask_trues, mask_preds)
                            if pred_level.shape[0] > 0])
        return summation / num_pos if num_pos != 0 else torch.tensor(0)
    
    def points_nms(self, heat, kernel=2):
        ''' 
        Credit to solo's auther: https://github.com/WXinlong/SOLO
        Input:
            - heat: (batch_size, C-1, S, S)
        Output:
                    (batch_size, C-1, S, S)
        '''
        # kernel must be 2
        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=1)
        keep = (hmax[:, :, :-1, :-1] == heat).float()
        return heat * keep

    def matrix_nms(self, sorted_masks, sorted_scores, method='gauss', gauss_sigma=0.5):
        ''' Performs Matrix NMS
        Input:
            - sorted_masks: (n_active, image_h/4, image_w/4)
            - sorted_scores: (n_active,)
        Output:
            - decay_scores: (n_active,)
        '''
        n = len(sorted_scores)
        sorted_masks = sorted_masks.reshape(n, -1)
        intersection = torch.mm(sorted_masks, sorted_masks.T)
        areas = sorted_masks.sum(dim=1).expand(n, n)
        union = areas + areas.T - intersection
        ious = (intersection / union).triu(diagonal=1)
    
        ious_cmax = ious.max(0)[0].expand(n, n).T
        if method == 'gauss':
            decay = torch.exp(-(ious ** 2 - ious_cmax ** 2) / gauss_sigma)
        else:
            decay = (1 - ious) / (1 - ious_cmax)
        decay = decay.min(dim=0)[0]
        return sorted_scores * decay

    def post_processing(self, cate_pred, mask_pred, nms_score_threshold = 0.1):

        cate_thresh = self.postprocess_cfg['cate_thresh']
        ins_thresh = self.postprocess_cfg['ins_thresh']
        pre_NMS_num = self.postprocess_cfg["pre_NMS_num"]
        keep_instance = self.postprocess_cfg['keep_instance']

        cate_flat = torch.cat([single[0].flatten() for single in cate_pred]).cpu()
        mask_flat = torch.cat([single[0] for single in mask_pred]).cpu()
        
        cate_score_all = cate_flat
        cate_all = torch.ones_like(cate_flat)
        mask_flat_all = mask_flat

        cate_score = cate_score_all[cate_score_all > cate_thresh][:pre_NMS_num]
        cate_label = cate_all[cate_score_all > cate_thresh][:pre_NMS_num]
        masks = mask_flat_all[cate_score_all > cate_thresh][:pre_NMS_num]

        binary_mask = (masks > ins_thresh).int()

        mask_score = torch.sum(torch.where(masks > ins_thresh, masks, torch.tensor(0.)), (1,2)) / torch.sum(binary_mask, (1,2))
        final_score = mask_score * cate_score

        sort_idx = torch.argsort(final_score, descending=True)
        sorted_binary_mask = binary_mask[sort_idx]
        sort_final_score = final_score[sort_idx]
        sorted_cate = cate_label[sort_idx]
        if len(sorted_binary_mask) < 1:
            return torch.zeros(0, *mask_flat.shape[-2:]), torch.tensor([]), torch.tensor([])

        nms_score = self.matrix_nms(sorted_binary_mask, sort_final_score)

        nms_sort_idx = torch.argsort(nms_score, descending=True)[:keep_instance]
        nms_sorted_final_score = nms_score[nms_sort_idx]
        nms_binary_mask = sorted_binary_mask[nms_sort_idx][nms_sorted_final_score > nms_score_threshold]
        nms_label = sorted_cate[nms_sort_idx][nms_sorted_final_score > nms_score_threshold]


        return nms_binary_mask, nms_label, sort_final_score[nms_sort_idx]
