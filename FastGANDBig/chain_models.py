import functools
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)


def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))


class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.target_shape = shape

    def forward(self, feat):
        batch = feat.shape[0]
        return feat.view(batch, *self.target_shape)


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out, bias=False):
        super().__init__()

        self.main = nn.Sequential(nn.AdaptiveAvgPool2d(4),
                                  conv2d(ch_in, ch_out, 4, 1, 0, bias=bias), Swish(),
                                  conv2d(ch_out, ch_out, 1, 1, 0, bias=bias), nn.Sigmoid())

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
            convTranspose2d(nz, channel * 2, 4, 1, 0, bias=False),
            batchNorm2d(channel * 2), GLU())

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def UpBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
        # convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        batchNorm2d(out_planes * 2), GLU())
    return block


def UpBlockComp(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
        # convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes * 2), GLU(),
        conv2d(out_planes, out_planes * 2, 3, 1, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes * 2), GLU()
    )
    return block


class Generator(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024, nosmall=True):
        super(Generator, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.im_size = im_size

        self.init = InitLayer(nz, channel=nfc[4])

        self.feat_8 = UpBlockComp(nfc[4], nfc[8])
        self.feat_16 = UpBlock(nfc[8], nfc[16])
        self.feat_32 = UpBlockComp(nfc[16], nfc[32])
        self.feat_64 = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlockComp(nfc[64], nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        self.se_64 = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        ###<modified>
        # self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
        self.nosmall = nosmall
        self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False) if not nosmall else None
        ###</modified>
        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False)

        if im_size > 256:
            self.feat_512 = UpBlockComp(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if im_size > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

    def forward(self, input):

        feat_4 = self.init(input)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)

        feat_64 = self.se_64(feat_4, self.feat_64(feat_32))

        feat_128 = self.se_128(feat_8, self.feat_128(feat_64))

        feat_256 = self.se_256(feat_16, self.feat_256(feat_128))

        if self.nosmall:
            if self.im_size == 256:
                return self.to_big(feat_256)

            feat_512 = self.se_512(feat_32, self.feat_512(feat_256))
            if self.im_size == 512:
                return self.to_big(feat_512)
            feat_1024 = self.feat_1024(feat_512)
            im_1024 = torch.tanh(self.to_big(feat_1024))
            return im_1024

        if self.im_size == 256:
            return [self.to_big(feat_256), self.to_128(feat_128)]

        feat_512 = self.se_512(feat_32, self.feat_512(feat_256))
        if self.im_size == 512:
            return [self.to_big(feat_512), self.to_128(feat_128)]

        feat_1024 = self.feat_1024(feat_512)

        im_128 = torch.tanh(self.to_128(feat_128))
        im_1024 = torch.tanh(self.to_big(feat_1024))

        return [im_1024, im_128]

class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""
    def __init__(self, nfc_in=64, nc=3, bias=False):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*32)

        def upBlock(in_planes, out_planes):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv2d(in_planes, out_planes*2, 3, 1, 1, bias=bias),
                batchNorm2d(out_planes*2), GLU())
            return block

        self.main = nn.Sequential(nn.AdaptiveAvgPool2d(8),
                                  upBlock(nfc_in, nfc[16]) ,
                                  upBlock(nfc[16], nfc[32]),
                                  upBlock(nfc[32], nfc[64]),
                                  upBlock(nfc[64], nfc[128]),
                                  conv2d(nfc[128], nc, 3, 1, 1, bias=bias),
                                  nn.Tanh())

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)


class CumulativeRMSNorm(torch.autograd.Function):
    """
    forward and backward function for cumulative root mean square normalization
    """
    @staticmethod
    def forward(ctx, x, decay, eps, running_var, grad_var):
        divisor = x.square().mean(axis=[0, 2, 3], keepdim=True)
        running_var.data.mul_(decay).add_(divisor, alpha=1 - decay)
        sp_norm = running_var.min().sqrt()
        divisor = (running_var + eps).sqrt()

        y = x / divisor
        ctx.decay = decay
        ctx.save_for_backward(y, divisor, sp_norm,  grad_var)
        return y * sp_norm

    @staticmethod
    def backward(ctx, grad_output):
        y, divisor, sp_norm, grad_var = ctx.saved_tensors
        decay = ctx.decay

        grad_output = grad_output * sp_norm
        grad_phi = (y * grad_output).mean(axis=[0, 2, 3], keepdim=True)
        grad_var.data.mul_(decay).add_(grad_phi, alpha=1-decay)
        grad_input = (grad_output - grad_var * y) / divisor
        return grad_input, None, None, None, None

class ChainSharedContext(nn.Module):
    """
    for managing p, and 0mr loss for Chain module
    """
    def __init__(self, tau, lbd=0., lbd_p0=0., delta_p=0.001, enable_update=False):
        super(ChainSharedContext, self).__init__()

        self.tau        = tau
        self.lbd        = lbd
        self.lbd_p0     = lbd_p0
        self.delta_p    = delta_p
        self.enable_update = enable_update

        self.loss_0mr = []
        self.label = 'dreal'

        self.register_buffer('p',         torch.zeros([], requires_grad=False, dtype=torch.float64))
        self.register_buffer('ema_Dreal', torch.zeros([], requires_grad=False, dtype=torch.float64))

    @torch.no_grad()
    def update_p(self, real_logits):
        """ this function is used for update p in the training loop
        :param real_logits: logits output by the discriminator for real samples.
        :return:
        """
        if not self.enable_update: return
        cur_Dreal = torch.sign(real_logits).mean()
        self.ema_Dreal.data.mul_(0.9).add_(cur_Dreal, alpha=0.1)
        self.p.data.add_(torch.sign(self.ema_Dreal - self.tau) * self.delta_p).clamp_(0., 1.)


    def reset(self, label, loss_0mr):
        self.loss_0mr.clear()
        self.loss_0mr.append(loss_0mr)
        self.label = label

    def get_p(self):
        return self.p.item()

    def get_loss_0mr(self):
        lbd = self.lbd * (self.get_p() + self.lbd_p0)
        if self.label == 'gfake': lbd = 0. #when training generator, 0mr loss is ignored, but you can uncomment to test if it helps improve performance.
        return torch.stack(self.loss_0mr).sum() * lbd

    def add_loss(self, loss):
        self.loss_0mr.append(loss)

class Chain(nn.Module):
    def __init__(self, num_features=None, shared_context=None,
                 chain_type=None,
                 decay=0.9, eps=1e-5):
        nn.Module.__init__(self)
        self.num_features = num_features
        if chain_type == 'chain':
            self.register_buffer('running_var_fake',    torch.ones(1, num_features, 1, 1, requires_grad=False))
            self.register_buffer('running_var_real',    torch.ones(1, num_features, 1, 1, requires_grad=False))
            self.register_buffer('grad_var_dfake',      torch.zeros(1, num_features, 1, 1, requires_grad=False))
            self.register_buffer('grad_var_gfake',      torch.zeros(1, num_features, 1, 1, requires_grad=False))
            self.register_buffer('grad_var_dreal',      torch.zeros(1, num_features, 1, 1, requires_grad=False))

        self.shared_context = shared_context
        self.chain_type = chain_type
        self.decay = decay
        self.eps = eps
        self.rmsn_func = CumulativeRMSNorm.apply
        self.forward_chain = None
        if chain_type == 'chain': self.forward_chain = self.Chain_forward
        elif chain_type == 'chain_batch': self.forward_chain = self.Chain_batch_forward

    def Chain_batch_forward(self, x, p):
        with torch.no_grad():
            M = (torch.rand(*x.shape[:2], 1, 1, device=x.device) <= p) * 1.
        psi_square = x.square().mean(axis=[0, 2, 3], keepdim=True)
        psi = (psi_square + self.eps).sqrt()
        psi_min = psi.min().detach()
        x_arms = (1 - M) * x + M * (x / psi * psi_min)
        return x_arms

    def Chain_forward(self, x, p):
        label = self.shared_context.label
        ### statistics for real or fake when training discriminator or generator is different
        if label == 'dfake':
            running_var = self.running_var_fake
            grad_var = self.grad_var_dfake
        elif label == 'dreal':
            running_var = self.running_var_real
            grad_var = self.grad_var_dreal
        elif label == 'gfake':
            running_var = self.running_var_fake
            grad_var = self.grad_var_gfake
        else:
            raise NotImplementedError

        with torch.no_grad():
            M = (torch.rand(*x.shape[:2], 1, 1, device=x.device) <= p) * 1.
        x_arms = (1 - M) * x + M * self.rmsn_func(x, self.decay, self.eps, running_var, grad_var)
        return x_arms

    def forward(self, x):
        if self.chain_type == None: return x

        p = self.shared_context.get_p()
        self.shared_context.add_loss(x.mean(axis=[0,2,3]).square().sum())

        if p == 0: return x

        return self.forward_chain(x, p)

class OptimizeBlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 bias       = False,
                 lrelu_slope=0.2,
                 ):
        super(OptimizeBlock, self).__init__()
        self.main = nn.Sequential(conv2d(in_planes, out_planes, 3, 1, 1, bias=bias), nn.LeakyReLU(lrelu_slope),
                                  conv2d(out_planes, out_planes, 3, 1, 1, bias=bias), nn.AvgPool2d(2, 2))

        self.skip = nn.Sequential(nn.AvgPool2d(2, 2), conv2d(in_planes, out_planes, 1, 1, 0, bias=bias))

    def forward(self, feat):
        return self.main(feat) + self.skip(feat)

class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes,
                 bias       = False,
                 downsample = True,
                 lrelu_slope=0.2,
                 chain_layer=functools.partial(Chain),
                 chain_place='',
                 ):
        super(DownBlockComp, self).__init__()
        self.main = nn.Sequential(
            nn.LeakyReLU(lrelu_slope, inplace=True),
            conv2d(in_planes, out_planes, 3, 1, 1, bias=bias),
            chain_layer(out_planes) if 'C1' in chain_place else chain_layer(chain_type=None),
            nn.LeakyReLU(lrelu_slope),
            conv2d(out_planes, out_planes, 3, 1, 1, bias=bias),
            chain_layer(out_planes) if 'C2' in chain_place else chain_layer(chain_type=None),
            nn.AvgPool2d(2, 2) if downsample else nn.Identity()
        )

        learnable_sc = downsample or in_planes != out_planes
        self.skip = nn.Sequential(
            conv2d(in_planes, out_planes, 1, 1, 0, bias=bias),
            chain_layer(out_planes) if 'CS' in chain_place else chain_layer(chain_type=None),
            nn.AvgPool2d(2, 2) if downsample else nn.Identity()
        ) if learnable_sc else nn.Identity()


    def forward(self, feat):
        return self.main(feat) + self.skip(feat)


class SumPool2d(nn.Module):
    def forward(self, x):
        return x.sum(dim=[2, 3])

class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512,
                 chain_type         =None,
                 chain_blocks       ='',
                 chain_place        ='',
                 tau                =0.,
                 lbd                =0.,
                 lbd_p0             =0.,
                 delta_p            =0.
                 ):
        super(Discriminator, self).__init__()
        self.ndf        = ndf
        self.im_size    = im_size

        nfc_multi = {4: 16, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ndf)

        blocks = [int(block_id) for block_id in chain_blocks]
        self.chain_shared_context = ChainSharedContext(tau, lbd, lbd_p0, delta_p, enable_update=chain_type != None)
        chain_layer = functools.partial(Chain, chain_type=chain_type, shared_context=self.chain_shared_context)
        chain_identity = functools.partial(Chain, chain_type=None)
        chain_layers = [chain_layer if i in blocks else chain_identity for i in range(7)]

        if im_size == 1024:
            self.down_from_big = nn.Sequential(conv2d(nc, nfc[1024], 4, 2, 1, bias=True),
                                               DownBlockComp(nfc[1024], nfc[512], bias=True))
        elif im_size == 512:
            self.down_from_big = conv2d(nc, nfc[512], 4, 2, 1, bias=True)
        elif im_size == 256:
            self.down_from_big = conv2d(nc, nfc[512], 3, 1, 1, bias=True)

        self.down_4 = DownBlockComp(nfc[512], nfc[256], bias=True, chain_layer=chain_layers[1], chain_place=chain_place)
        self.down_8 = DownBlockComp(nfc[256], nfc[128], bias=True, chain_layer=chain_layers[2], chain_place=chain_place)
        self.down_16 = DownBlockComp(nfc[128], nfc[64], bias=True, chain_layer=chain_layers[3], chain_place=chain_place)
        self.down_32 = DownBlockComp(nfc[64], nfc[32], bias=True, chain_layer=chain_layers[4], chain_place=chain_place)
        self.down_64 = DownBlockComp(nfc[32], nfc[16], bias=True, chain_layer=chain_layers[5], chain_place=chain_place)

        self.rf_big = nn.Sequential(DownBlockComp(nfc[16], nfc[16], bias=True, downsample=False),
                                    SumPool2d(), linear(nfc[16], 1, bias=True))

        self.se_2_16 = SEBlock(nfc[512], nfc[64])
        self.se_4_32 = SEBlock(nfc[256], nfc[32])
        self.se_8_64 = SEBlock(nfc[128], nfc[16])

        self.decoder_big = SimpleDecoder(nfc[16], nc)
        self.decoder_part = SimpleDecoder(nfc[32], nc)

    def forward(self, imgs, label, part=None):
        self.chain_shared_context.reset(label, torch.zeros([], device=imgs.device))

        feat_2 = self.down_from_big(imgs)
        feat_4 = self.down_4(feat_2)
        feat_8 = self.down_8(feat_4)
        feat_16 = self.down_16(feat_8)
        feat_16 = self.se_2_16(feat_2, feat_16)
        feat_32 = self.down_32(feat_16)
        feat_32 = self.se_4_32(feat_4, feat_32)
        feat_last = self.down_64(feat_32)
        feat_last = self.se_8_64(feat_8, feat_last)

        rf = self.rf_big(feat_last).view(-1)
        loss_0mr = self.chain_shared_context.get_loss_0mr()

        if label == 'dreal':
            rec_img_big = self.decoder_big(feat_last)
            assert part is not None
            rec_img_part = None
            if part == 0:
                rec_img_part = self.decoder_part(feat_32[:, :, :8, :8])
            if part == 1:
                rec_img_part = self.decoder_part(feat_32[:, :, :8, 8:])
            if part == 2:
                rec_img_part = self.decoder_part(feat_32[:, :, 8:, :8])
            if part == 3:
                rec_img_part = self.decoder_part(feat_32[:, :, 8:, 8:])
            return rf, [rec_img_big, rec_img_part], loss_0mr
        return rf, loss_0mr

