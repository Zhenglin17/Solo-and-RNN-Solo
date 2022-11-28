\
# %%
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from tqdm import tqdm, trange
import random
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch import nn 
import os
from IPython.display import clear_output
import timeit
import ast

import warnings
warnings.filterwarnings("ignore")

store_folder = input('checkpoints folder name: ')

checkpoint_path = f'../check_points/{store_folder}'

if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)
    
print('Checkpoints path : ', checkpoint_path)
# %%
draw_box = lambda x1, y1, x2, y2: np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]])

# %% [markdown]
# # Load Data 
if_load = input('Load presets? T/F')[0]
if  if_load in ['t', 'T']:
    from config import *
else:
# %%
    num_grids = input('num_grids:')
    num_grids = ast.literal_eval(num_grids)
    num_grids = [int(f) for f in num_grids]

                    
    scale_ranges = input('scale_ranges: ')
    scale_ranges = ast.literal_eval(scale_ranges)
    scale_ranges = [[float(e) for e in f] for f in scale_ranges]
    batch_size = int(input('batch_size: '))
    mask_weight = float(input("mask_weight: "))
    cate_weight = float(input("cate_weight: "))
    eps = float(input("eps: "))
from data_loader import *
DATA_DIR = "../processdata/"
train_files, test_files = train_test_split(DATA_DIR)
test_files = list(np.load("test_filenames.npy"))
train_files = list(np.load("train_filenames.npy"))
batch_size = 4
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Grayscale(),
                                                torchvision.transforms.Normalize([0.5], [0.5])])
train_loader = recurrent_loader(train_files, DATA_DIR, batch_size = batch_size,transforms = transforms, lags = 2)
test_loader = recurrent_loader(test_files, DATA_DIR, batch_size = batch_size,transforms = transforms, lags = 2)

# %% [markdown]
# ## Visual Inspection

# %%
# data = next(iter(train_loader))
for data in train_loader:
    break
sequences, masks, bboxes, batch = data



# %%
from architecture import *
postprocess_cfg = dict(cate_thresh=0.5,
                            ins_thresh=0.5,
                            pre_NMS_num=100,
                            # pre_NMS_num=50,
                            keep_instance=10)
                            # keep_instance=15)

# %%


# %% [markdown]
# ## SOLO

# %%
class SOLO(SOLO_head):    
    def __init__(self,
                 scale_ranges=scale_ranges,
                 num_grids=num_grids, 
                 mask_loss_cfg=dict(weight=mask_weight),
                 cate_loss_cfg=dict(gamma=2,
                                    alpha=0.25,
                                    weight=cate_weight),
                 postprocess_cfg=postprocess_cfg):  
        super(SOLO, self).__init__(scale_ranges, num_grids, mask_loss_cfg, cate_loss_cfg, postprocess_cfg)
    
    
        self.backbone = FPN(Bottleneck, [3,4,6,3])
        self.grus = nn.ModuleList([])

        for i in range(4):
            self.grus.append(bi_GRU(256,256,3))
    
    def forward(self, sequences, device, eval=False, original_img = None):

        sequences_feature = self.backbone.sequence_output(sequences)
        feature_pyramid = []

        for i in range(4):
            feature_pyramid.append(self.grus[i](sequences_feature[i]))



            

        category_predictions = []
        mask_predictions = []

        if eval == True:
            
            for i, v in enumerate(feature_pyramid):
                num_grid = self.num_grids[i]
                category_pred = nn.functional.interpolate(v, size=[num_grid, num_grid], mode="bilinear", align_corners = True)
                category_pred = self.cate_branch(category_pred)
                category_predictions.append(self.points_nms(category_pred).permute(0,2,3,1))

                meshxy = torch.stack(torch.meshgrid(torch.linspace(-1,1,v.shape[-2]), torch.linspace(-1,1,v.shape[-1])))
                batch_meshxy = torch.tile(meshxy, (sequences.shape[0],1,1,1)).to(device)
                new_v = torch.cat((v, batch_meshxy), 1)

                mask_pred = self.ins_outs[i](self.ins_branch(new_v))

                mask_predictions.append(nn.functional.interpolate(mask_pred, size=original_img, mode='bilinear', align_corners = True))
            return category_predictions, mask_predictions         
            
        else:
            for i, v in enumerate(feature_pyramid):
                num_grid = self.num_grids[i]
                category_pred = nn.functional.interpolate(v, size=[num_grid, num_grid], mode="bilinear", align_corners = True)
                category_predictions.append(self.cate_branch(category_pred))

                meshxy = torch.stack(torch.meshgrid(torch.linspace(-1,1,v.shape[-2]), torch.linspace(-1,1,v.shape[-1])))
                batch_meshxy = torch.tile(meshxy, (sequences.shape[0],1,1,1)).to(device)
                new_v = torch.cat((v, batch_meshxy), 1)

                mask_pred = self.ins_outs[i](self.ins_branch(new_v))

                mask_predictions.append(nn.functional.interpolate(mask_pred, scale_factor=2, mode='bilinear', align_corners = True))
            return category_predictions, mask_predictions 

# %%
device = 'cuda:0'
model = SOLO().to(device)
backbone_out = model.backbone(torch.zeros(0,1,313,313).to(device))
model.featmap_sizes = [list(f.shape[-2:]) for f in backbone_out]
labels = [torch.ones(b.shape[0]) for b in bboxes]
masks, bboxes = [torch.Tensor(m) for m in masks], [torch.Tensor(b) for b in bboxes]
target = model.generate_single_target(bboxes[0], labels[0], masks[0], model.featmap_sizes, eps = eps)

# %%
targets = model.generate_targets(masks, bboxes, labels, eps = eps, device = device)
out = model(sequences.to(device), device = device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
np.save(f'{checkpoint_path}/hyper_parameters.npy', {'num_grids': num_grids, 'scale_ranges': scale_ranges, "eps":eps, "optimizer" : type(optimizer).__name__, 'mask_weight':mask_weight, 'cate_weight':cate_weight}, allow_pickle = True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 60, 75], gamma=0.95)

num_epoch = 75
print(f"batch size : {batch_size}, epoch : {num_epoch}")
loss_train_stor = []
loss_val_stor = []


model.postprocess_cfg = postprocess_cfg


for epoch in tqdm(range(num_epoch)):
    
    batch = 0
    n_show = 3
    model.eval()
    for i, data in enumerate(train_loader):
    # for i, data in enumerate(test_loader):

        sequences, masks, bboxes, _ = data
        masks, bboxes = [torch.Tensor(m) for m in masks], [torch.Tensor(b) for b in bboxes]
        original_img= [int(313/2),int(313/2)]

        post_out = model.post_processing(*model.forward(sequences.to(device),
                                                        device=device, 
                                                        eval=True, 
                                                        original_img=original_img),
                                        nms_score_threshold = 5e-1)

        plt.figure(figsize = (18,6))

        plt.subplot(1, 3, 1)
        plt.title('Raw Image')
        plt.imshow(sequences[batch][0][-1].squeeze(), cmap = 'gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title(f'Ground Truth: {len(bboxes[batch])} instances')
        plt.imshow(sequences[batch][0][-1].squeeze(), cmap = 'gray')
        for box, msk in zip(bboxes[batch], masks[batch]):
            plt.plot(*draw_box(*box * 313).T, color = 'aqua')
            plt.imshow(msk, alpha = msk * 0.5, cmap = 'Reds')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title(f'Prediction: {len(post_out[batch])} instances')
        plt.imshow(sequences[batch][0][-1].squeeze(), cmap = 'gray')
        for msk in post_out[0]:
            m = (nn.functional.interpolate(msk[None, None].float(), size=sequences.shape[-2:], mode='bilinear') > 0.5)[0,0].int().cpu().numpy()
            plt.imshow(m, alpha = m * 0.5, cmap = 'winter')
        plt.axis('off')
        if i == n_show-1:
            plt.savefig(f'{checkpoint_path}/solo-epoch-{epoch}-train-result.pdf', dpi = 100)
        plt.show()
        plt.close()

        if i == n_show-1:
            break

    # plt.savefig(f'{checkpoint_path}/train_outputs_epoch_{epoch + 1}.pdf', dpi = 100)

    
    model.train()
    print(f'Eps : {eps:.1f}')
    print({'num_grids': num_grids, 'scale_ranges': scale_ranges, "eps":eps, 'cate_weight':cate_weight, 'mask_weight' : mask_weight})

    print(f'\n== EPOCH {epoch+1}; Loss:  Cate      Mask      Total  Elapsed time')

    # Train
    start_time = timeit.default_timer()
    for iter, data in enumerate(train_loader):

        optimizer.zero_grad()
        
        # load data and adjust dtype
        sequences, masks, bboxes, batch = data
        masks, bboxes = [torch.Tensor(m) for m in masks], [torch.Tensor(b) for b in bboxes]
        sequences = sequences.to(device)
        labels = [torch.ones(b.shape[0]).to(device) for b in bboxes]

        targets = model.generate_targets(masks, bboxes, labels, eps = eps, device = device)
        out = model(sequences.to(device), device = device)
        cate_loss_train, mask_loss_train, loss_train = model.loss(*out, *targets, device)
        loss_train_stor.append((cate_loss_train.cpu().detach().item(), 
                                mask_loss_train.cpu().detach().item(), 
                                loss_train.cpu().detach().item()))
        loss_train.backward()
        optimizer.step()
        if iter % 10 == 0:
            print(f"Train Iter: {iter:>3}; {cate_loss_train.cpu().detach().item():.2e}, {mask_loss_train.cpu().detach().item():.2e}, {loss_train.cpu().detach().item():.2e}, {timeit.default_timer() - start_time:.1f}")
    
    scheduler.step()
    start_time = timeit.default_timer()
    
    with torch.no_grad():
        model.eval()
    
        for iter, data in enumerate(test_loader):

            optimizer.zero_grad()

            # load data and adjust dtype
            sequences, masks, bboxes, batch = data
            masks, bboxes = [torch.Tensor(m) for m in masks], [torch.Tensor(b) for b in bboxes]
            sequences = sequences.to(device)
            labels = [torch.ones(b.shape[0]).to(device) for b in bboxes]

            targets = model.generate_targets(masks, bboxes, labels, eps = eps, device = device)
            out = model(sequences.to(device), device = device)
            cate_loss_val, mask_loss_val, loss_val = model.loss(*out, *targets, device)
            loss_val_stor.append((cate_loss_val.cpu().detach().item(), 
                                    mask_loss_val.cpu().detach().item(), 
                                    loss_val.cpu().detach().item()))
            
            if iter % 10 == 0:
                print(f"Test Iter: {iter:>3}; {cate_loss_val.cpu().detach().item():.2e}, {mask_loss_val.cpu().detach().item():.2e}, {loss_val.cpu().detach().item():.2e}, {timeit.default_timer() - start_time:.1f}")

    path = f'{checkpoint_path}/solo-epoch-{epoch+1}.pth'
    torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, path)
    

    np.save(f'{checkpoint_path}/loss-train-epoch-{epoch+1}.npy', np.array(loss_train_stor))
    np.save(f'{checkpoint_path}/loss-val-epoch-{epoch+1}.npy', np.array(loss_val_stor))
    
    if epoch+1 < num_epoch: # not the last epoch
        clear_output()
