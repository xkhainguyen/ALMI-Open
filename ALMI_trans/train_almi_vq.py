import os
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import math

import models.vqvae as vqvae
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_OL_VQ

import warnings
warnings.filterwarnings('ignore')

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- Dataloader ---- #####
train_dataset = dataset_OL_VQ.ALMIVQMotionDataset(args.dataname, window_size=args.window_size)
batch_sampler = dataset_OL_VQ.VQBatchSampler(train_dataset, padding_len=0,)
train_loader = torch.utils.data.DataLoader(
                        dataset=train_dataset,
                        batch_size=args.batch_size, # 32
                        sampler=batch_sampler,
                        num_workers=8,
                        # persistent_workers=(num_workers > 0),
                        pin_memory=torch.cuda.is_available(),)

all_motion_len = len(batch_sampler)
one_epoch_iters = math.ceil(all_motion_len / args.batch_size)
print(f'one epoch iters: {one_epoch_iters}')

##### ---- Network ---- #####
net = vqvae.VQVAE(args,
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate,
                       args.vq_act,
                       args.vq_norm,)


if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
net.train()
net.to(args.device)

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

Loss = torch.nn.SmoothL1Loss()

##### ------ warm-up ------- #####
avg_recons, avg_perplexity, avg_commit, nb_epoch = 0., 0., 0., 0
train_loader_iter = iter(train_loader)
for nb_iter in range(1, args.warm_up_iter):
    
    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
    
    gt_motion = next(train_loader_iter)
    gt_motion = gt_motion.to(args.device).float() # (bs, 64, dim)

    pred_motion, loss_commit, perplexity = net(gt_motion)
    loss_motion = Loss(pred_motion, gt_motion)
    
    loss = loss_motion + args.commit * loss_commit # + args.loss_vel * loss_vel
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_recons += loss_motion.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    
    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        
        logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
        
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

##### ---- Training ---- #####
avg_recons, avg_perplexity, avg_commit, nb_iter, nb_epoch = 0., 0., 0., 0, 0

while nb_epoch <= args.total_epoch:
    if nb_epoch == 0:
        torch.save({'net_vq' : net.state_dict()}, os.path.join(args.out_dir, f'vq_net_{nb_epoch}.pth'))
        logger.info(f'model saved in epoch {nb_epoch}')
    train_loader_iter = iter(train_loader)
    nb_epoch += 1
    for i in range(one_epoch_iters):
        gt_motion = next(train_loader_iter)
        gt_motion = gt_motion.to(args.device).float() # bs, nb_joints, joints_dim, seq_len
        
        pred_motion, loss_commit, perplexity = net(gt_motion)
        loss_motion = Loss(pred_motion, gt_motion)
        
        loss = loss_motion + args.commit * loss_commit  # + args.loss_vel * loss_vel
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        avg_recons += loss_motion.item()
        avg_perplexity += perplexity.item()
        avg_commit += loss_commit.item()
        
        nb_iter += 1
        if nb_iter % args.print_iter ==  0 :
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter
            
            writer.add_scalar('./Train/L1', avg_recons, nb_iter)
            writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
            writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
            
            logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
            
            avg_recons, avg_perplexity, avg_commit = 0., 0., 0.,

    torch.save({'net_vq' : net.state_dict()}, os.path.join(args.out_dir, 'vq_net_last.pth'))
    logger.info(f'model last saved {nb_epoch}')
