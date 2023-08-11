import os
import time
import argparse

import torchvision
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('../../')
from util import AverageMeter, MultiAugUnsupervisedDataset, Imbalanced_CIFAR10
from encoder import SmallAlexNet
from losses import our_loss1, our_loss2


method_name = 'ours'

def parse_option():
    parser = argparse.ArgumentParser('CIFAR10 Representation Learning with Our Losses')
    # CACR options
    parser.add_argument('--Ny', type=int, default=4, help='positive samples for each sample in each batch')
    parser.add_argument('--Ns', type=int, default=128, help='Note that Ns-1 is exactly the meaning in our paper, which means sampling Ns-1 negative samples for each sample in each batch. You can also regard this Ns here as the batch size')
    parser.add_argument('--tau_pos', type=float, default=1.0, help='temperature for conditional positive weight calculation')
    parser.add_argument('--tau_neg', type=float, default=1.0, help='temperature for conditional negative weight calculation')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha for positive loss calculation')
    parser.add_argument('--beta', type=float, default=1.0, help='beta for negative loss calculation')
    parser.add_argument('--choice', type=str, default='full', help='run choice for our loss, full, without_w1, without_w2, none')
    parser.add_argument('--tau_plus', type=float, default=0.1, help='tau+ for debiased version, only works when with_debiased=True and choice=full')
    parser.add_argument('--with_debiased', action='store_true', help='whether combine debiased to calculate the loss')
    # training options
    parser.add_argument('--data_folder', type=str, default='./data', help='Path to data')
    parser.add_argument('--imbalanced', action='store_true', help='whether use imbalanced datasets to train the model')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate. Default is linear scaling 0.12 per 256 batch size')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--lr_decay_epochs', default=[155, 170, 185], nargs='*', type=int,
                        help='When to decay learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 weight decay')
    parser.add_argument('--feat_dim', type=int, default=128, help='Feature dimensionality')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of data loader workers to use')
    parser.add_argument('--log_interval', type=int, default=40, help='Number of iterations between logs')
    parser.add_argument('--gpus', default=[3, 0], nargs='*', type=int,
                        help='List of GPU indices to use, e.g., --gpus 0 1 2 3')
    # saving options
    parser.add_argument('--result_folder', type=str, default='./results', help='Base directory to save model')
    parser.add_argument('--log_folder', type=str, default='./logs', help='Base directory to save logs')

    opt = parser.parse_args()
    # Note that we directly set batch_size=Ns here for simplicity, which means the intra-batch negative sampling size is Ns-1
    opt.batch_size = opt.Ns

    if opt.lr is None:
        opt.lr = 0.12 * (opt.batch_size / 256)

    opt.gpus = list(map(lambda x: torch.device('cuda', x), opt.gpus))

    opt.save_folder = os.path.join(
        opt.result_folder, method_name,
        f'Ny{opt.Ny}_Ns{opt.Ns}',
        f'tau_pos{opt.tau_pos}_tau_neg{opt.tau_neg}_alpha{opt.alpha}_beta{opt.beta}_choice_{opt.choice}_with_debiased_{opt.with_debiased}_with_imbalanced_{opt.imbalanced}',
    )
    opt.log_path = os.path.join(
        opt.log_folder, method_name,
        f'Ny{opt.Ny}_Ns{opt.Ns}',
        f'tau_pos{opt.tau_pos}_tau_neg{opt.tau_neg}_alpha{opt.alpha}_beta{opt.beta}_choice_{opt.choice}_with_debiased_{opt.with_debiased}_with_imbalanced_{opt.imbalanced}',
    )
    os.makedirs(opt.save_folder, exist_ok=True)
    if os.path.exists(opt.log_path):
        os.system('rm -r {}'.format(opt.log_path))
    else:
        os.makedirs(opt.log_path, exist_ok=True)

    return opt


def get_data_loader(opt):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(32, scale=(0.2, 1)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])
    if not opt.imbalanced:
        dataset = MultiAugUnsupervisedDataset(
            torchvision.datasets.CIFAR10(opt.data_folder, train=True, download=True), transform=transform, Ny=opt.Ny+1)
    else:
        dataset = MultiAugUnsupervisedDataset(
            Imbalanced_CIFAR10(opt.data_folder, train=True, download=True), transform=transform, Ny=opt.Ny+1)
            
    return torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                       shuffle=True, pin_memory=True)


def main():
    opt = parse_option()
    writer = SummaryWriter(opt.log_path)

    print(f'Optimize: Ny_(Ny={opt.Ny})+Ns{opt.Ns}')

    torch.cuda.set_device(opt.gpus[0])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    encoder = nn.DataParallel(SmallAlexNet(feat_dim=opt.feat_dim).to(opt.gpus[0]), opt.gpus)

    optim = torch.optim.SGD(encoder.parameters(), lr=opt.lr,
                            momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=opt.lr_decay_rate,
                                                     milestones=opt.lr_decay_epochs)

    loader = get_data_loader(opt)

    loss1_meter = AverageMeter('loss1')
    loss2_meter = AverageMeter('loss2')
    loss_meter = AverageMeter('total_loss')
    it_time_meter = AverageMeter('iter_time')
    for epoch in range(opt.epochs):
        loss1_meter.reset()
        loss2_meter.reset()
        loss_meter.reset()
        it_time_meter.reset()
        t0 = time.time()
        for ii, samples_list in enumerate(loader):
            optim.zero_grad()
            latent_codes_list = []
            for j in range(opt.Ny+1):
                feed_input = samples_list[j].to(opt.gpus[j % len(opt.gpus)])
                latent_codes = encoder(feed_input)
                latent_codes_list.append(latent_codes)
                del feed_input
            loss1_val, weights1 = our_loss1(latent_codes_list, alpha=opt.alpha, tau_pos=opt.tau_pos, choice=opt.choice) # compute positive transport
            loss2_val, weights2 = our_loss2(latent_codes_list, beta=opt.beta, tau_neg=opt.tau_neg, choice=opt.choice, with_debiased=opt.with_debiased, tau_plus=opt.tau_plus) # compute negative transport
            loss = loss1_val + loss2_val
            loss1_meter.update(loss1_val, latent_codes_list[0].shape[0])
            loss2_meter.update(loss2_val)
            loss_meter.update(loss, latent_codes_list[0].shape[0])
            loss.backward()
            optim.step()
            it_time_meter.update(time.time() - t0)
            if ii % opt.log_interval == 0:
                print(f"Epoch {epoch}/{opt.epochs}\tIt {ii}/{len(loader)}\t" +
                      f"{loss1_meter}\t{loss2_meter}\t{loss_meter}\t{it_time_meter}")
            t0 = time.time()
        scheduler.step()
        writer.add_scalar('train/loss1', loss1_meter.avg, epoch)
        writer.add_scalar('train/loss2', loss2_meter.avg, epoch)
        writer.add_scalar('train/loss', loss_meter.avg, epoch)
    ckpt_file = os.path.join(opt.save_folder, 'encoder.pth')
    torch.save(encoder.module.state_dict(), ckpt_file)
    print(f'Saved to {ckpt_file}')


if __name__ == '__main__':
    main()
