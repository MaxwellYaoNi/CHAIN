import time

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

import argparse
import random
from tqdm import tqdm

from operation import copy_G_params, load_params
from operation import ImageFolder, InfiniteSamplerWrapper
from operation import get_dir, seed_rng, MetricsLogger, MinSizeCenterCrop, EasyDict
from torch_inception_metrics import TorchInceptionMetrics
from chain_models import weights_init, Discriminator, Generator
from torchvision import utils as vutils

from diffaug import DiffAugment

import lpips, os
seed_rng(0)
torch.hub.set_dir('cache')
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=torch.cuda.is_available())


def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]


def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label=="dreal":
        part = random.randint(0, 3)
        pred, [rec_all, rec_part], reg_loss = net(data, label, part=part)
        err = F.relu(1. - pred).mean() + reg_loss
        err += percept(rec_all, F.interpolate(data, rec_all.shape[2])).sum() + \
               percept(rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2])).sum()
        err.backward()
        return pred.mean().item(), rec_all, rec_part, pred, reg_loss
    else:
        pred, reg_loss = net(data, label)
        err = F.relu(1. + pred).mean() + reg_loss
        err.backward()
        return pred.mean().item(), reg_loss



def train(config):
    data_root           = config.path
    total_iterations    = config.iter
    batch_size          = config.batch_size
    im_size             = config.im_size
    ndf                 = config.ndf
    ngf                 = config.ngf
    multi_gpu           = config.multi_gpu
    save_interval       = config.save_interval
    nz                  = 256
    nlr                 = 0.0002
    dataloader_workers  = 8

    policy              = 'color,translation'

    saved_model_folder, saved_image_folder = get_dir(config)
    seed_rng(config.seed)
    inception_batch_size = 64 * max(1, torch.cuda.device_count())
    metrics_calculator = TorchInceptionMetrics(config.inception_path, batch_size=inception_batch_size, torch_for_fid=config.torchfid, eps=1e-6)
    metrics_calculator.load_train_fid_moments(config.moments_path, verbose=True)
    test_logger = MetricsLogger(config.metrics_log_path, reinitialize=True)

    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    transform_list = []
    if config.center_crop: transform_list.append(MinSizeCenterCrop())
    transform_list += [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)


    
    if 'lmdb' in data_root:
        from operation import MultiResolutionDataset
        dataset = MultiResolutionDataset(data_root, trans, 1024)
    else:
        dataset = ImageFolder(root=data_root, transform=trans)

   
    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))
    '''
    loader = MultiEpochsDataLoader(dataset, batch_size=batch_size, 
                               shuffle=True, num_workers=dataloader_workers, 
                               pin_memory=True)
    dataloader = CudaDataLoader(loader, 'cuda')
    '''
    
    
    #from model_s import Generator, Discriminator
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, im_size=im_size,
                         chain_type     =config.chain_type,
                         chain_blocks   =config.chain_blocks,
                         chain_place    =config.chain_place,
                         tau            =config.tau,
                         lbd            =config.lbd,
                         lbd_p0         =config.lbd_p0,
                         delta_p        =config.delta_p)
    print(netD)
    netD.apply(weights_init)

    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(32, nz).normal_(0, 1).to(device).split(batch_size)
    
    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(0.5, 0.999))
        
    if multi_gpu:
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))

    netG.train()
    netD.train()

    def sample_func():
        noise_z = torch.Tensor(batch_size, nz).normal_(0, 1).to(device)
        samples = netG(noise_z).add(1).mul(0.5)
        return samples

    for iteration in tqdm(range(total_iterations+1)):
        real_image = next(dataloader)
        real_image = real_image.to(device)
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)

        fake_image = netG(noise)

        real_image = DiffAugment(real_image, policy=policy, random_order=True)
        fake_image = DiffAugment(fake_image, policy=policy, random_order=True)

        ## 2. train Discriminator
        netD.zero_grad()
        err_dr, _, _, real_logits, real_reg = train_d(netD, real_image, label="dreal")
        _, fake_reg = train_d(netD, fake_image.detach(), label="dfake")
        optimizerD.step()

        ### update p in Chain
        if iteration >= config.p_start_iter:
            netD.chain_shared_context.update_p(real_logits)
        
        ## 3. train Generator
        netG.zero_grad()
        pred_g, _ = netD(fake_image, "gfake")
        err_g = -pred_g.mean()
        err_g.backward()
        optimizerG.step()

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 100 == 0:
            print("GAN: loss d: %.5f  loss g: %.5f, real_reg:%.4f, fake_reg:%.4f\n"%(err_dr, -err_g.item(), real_reg.item(), fake_reg.item()))
          
        if iteration > 0 and iteration % save_interval == 0 or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                FID = metrics_calculator.sample_and_FID(sample_func, config.num_test_images)
                sample_images = torch.cat([netG(fixed_z).add(1).mul(0.5) for fixed_z in fixed_noise])
                vutils.save_image(sample_images, os.path.join(saved_image_folder, f'fake_{iteration}.jpg'), nrow=4, padding=0)

            load_params(netG, backup_para)
            with torch.no_grad():
                torch.save({'g': netG.state_dict(), 'd': netD.state_dict(), 'g_ema': avg_param_G,
                            'opt_g': optimizerG.state_dict(), 'opt_d': optimizerD.state_dict()},
                           os.path.join(saved_model_folder, 'network.pth'))
            test_logger.log(iter=iteration, FID=FID)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CHAIN FastGANDBig')
    parser.add_argument('--path', type=str, default='../lmdbs/art_landscape_1k',
                        help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--iter',               type=int, default=100000, help='number of iterations')
    parser.add_argument('--batch_size',         type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size',            type=int, default=1024, help='image resolution')

    parser.add_argument('--out_dir',            type=str, default='outs')
    parser.add_argument('--multi_gpu',          type=int, default=0)
    parser.add_argument('--dataset_name',       type=str, default='')
    parser.add_argument('--ndf',                type=int, default=64)
    parser.add_argument('--ngf',                type=int, default=64)
    parser.add_argument('--save_interval',      type=int, default=1000)
    parser.add_argument('--moments_path',       type=str, default='')
    parser.add_argument('--seed',               type=int, default=0)
    parser.add_argument('--inception_path',     type=str, default='data/inception_model')
    parser.add_argument('--num_test_images',    type=int, default=5000)
    parser.add_argument('--torchfid',           action='store_true', default=False)
    parser.add_argument('--center_crop',        action='store_true', default=False)

    parser.add_argument('--chain_type',         type=str,   default=None, choices=[None, 'chain', 'chain_batch'],
                        help='chain: cumulative chain, chain_batch: chain batch')
    parser.add_argument('--chain_blocks',       type=str,   default='', help='discriminator blocks applying chain')
    parser.add_argument('--chain_place',        type=str,   default='C1C2CS', help='apply the chain after convolution C1, C2, or CS')
    parser.add_argument('--lbd',                type=float, default=20)
    parser.add_argument('--lbd_p0',             type=float, default=0., help='strength of 0mr will be: lbd * (p + self.lbd_p0)')
    parser.add_argument('--tau',                type=float, default=0.5)
    parser.add_argument('--delta_p',            type=float, default=0.001)
    parser.add_argument('--p_start_iter',       type=int,   default=5000, help='iteration starts to evaluate whether discriminator is overfitting')


    config = EasyDict(vars(parser.parse_args()))
    print(config)
    train(config)
