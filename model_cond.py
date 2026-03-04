import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi
import numpy as np
from scipy import linalg as la

logabs = lambda x: torch.log(torch.abs(x) + 1e-12)


# -------------------------
# FiLM conditioning
# -------------------------
class FiLM(nn.Module):
    """
    Given cond: [B, cond_dim] and feature map h: [B, C, H, W]
    gamma,beta -> [B,C,1,1]
    h = h * (1 + gamma) + beta
    """
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_gb = nn.Linear(cond_dim, 2 * channels)
        # start near identity
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)

    def forward(self, h, cond):
        gb = self.to_gb(cond)  # [B,2C]
        gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return h * (1.0 + gamma) + beta


# -------------------------
# Glow components
# -------------------------
class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, x):
        with torch.no_grad():
            # flatten per-channel across batch+spatial
            flatten = x.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
            mean = flatten.mean(1).view(1, -1, 1, 1)
            std = flatten.std(1).view(1, -1, 1, 1)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, x):
        B, _, H, W = x.shape
        if self.initialized.item() == 0:
            self.initialize(x)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)
        logdet = (H * W * torch.sum(log_abs)) * torch.ones(B, device=x.device)
        y = self.scale * (x + self.loc)
        return (y, logdet) if self.logdet else y

    def reverse(self, y):
        return y / self.scale - self.loc


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))

        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)

        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        self.register_buffer("w_p", torch.from_numpy(w_p.copy()))
        self.register_buffer("u_mask", torch.from_numpy(u_mask.copy()))
        self.register_buffer("l_mask", torch.from_numpy(l_mask.copy()))
        self.register_buffer("s_sign", torch.sign(torch.from_numpy(w_s.copy())))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))

        self.w_l = nn.Parameter(torch.from_numpy(w_l.copy()))
        self.w_s = nn.Parameter(logabs(torch.from_numpy(w_s.copy())))
        self.w_u = nn.Parameter(torch.from_numpy(w_u.copy()))

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight.unsqueeze(2).unsqueeze(3)

    def forward(self, x):
        B, _, H, W = x.shape
        weight = self.calc_weight()
        y = F.conv2d(x, weight)
        logdet = (H * W * torch.sum(self.w_s)) * torch.ones(B, device=x.device)
        return y, logdet

    def reverse(self, y):
        weight = self.calc_weight()
        inv = weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        return F.conv2d(y, inv)


class ZeroConv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

    def forward(self, x):
        x = F.pad(x, [1, 1, 1, 1], value=1)
        x = self.conv(x)
        x = x * torch.exp(self.scale * 3)
        return x


# -------------------------
# Conditional nets
# -------------------------
class CondCouplingNet(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, hidden=512):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.f1 = FiLM(cond_dim, hidden)

        self.c2 = nn.Conv2d(hidden, hidden, 1)
        self.f2 = FiLM(cond_dim, hidden)

        self.c3 = ZeroConv2d(hidden, out_ch)

        self.c1.weight.data.normal_(0, 0.05)
        self.c1.bias.data.zero_()
        self.c2.weight.data.normal_(0, 0.05)
        self.c2.bias.data.zero_()

    def forward(self, x, cond):
        h = self.c1(x)
        h = self.f1(h, cond)
        h = F.relu(h, inplace=True)

        h = self.c2(h)
        h = self.f2(h, cond)
        h = F.relu(h, inplace=True)

        return self.c3(h)


class CondPriorNet(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, in_ch, 1)
        self.f1 = FiLM(cond_dim, in_ch)
        self.c2 = ZeroConv2d(in_ch, out_ch)

        self.c1.weight.data.normal_(0, 0.05)
        self.c1.bias.data.zero_()

    def forward(self, x, cond):
        h = self.c1(x)
        h = self.f1(h, cond)
        h = F.relu(h, inplace=True)
        return self.c2(h)


class AffineCouplingCond(nn.Module):
    def __init__(self, in_channel, cond_dim, filter_size=512, affine=True):
        super().__init__()
        self.affine = affine
        out_ch = in_channel if affine else in_channel // 2
        self.net = CondCouplingNet(in_channel // 2, out_ch, cond_dim, hidden=filter_size)

    def forward(self, x, cond):
        xa, xb = x.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(xa, cond).chunk(2, 1)
            s = torch.sigmoid(log_s + 2)
            yb = (xb + t) * s
            logdet = torch.sum(torch.log(s).view(x.shape[0], -1), 1)
        else:
            yb = xb + self.net(xa, cond)
            logdet = None
        return torch.cat([xa, yb], 1), logdet

    def reverse(self, y, cond):
        ya, yb = y.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(ya, cond).chunk(2, 1)
            s = torch.sigmoid(log_s + 2)
            xb = yb / s - t
        else:
            xb = yb - self.net(ya, cond)
        return torch.cat([ya, xb], 1)


class FlowCond(nn.Module):
    def __init__(self, in_channel, cond_dim, affine=True):
        super().__init__()
        self.actnorm = ActNorm(in_channel)
        self.invconv = InvConv2dLU(in_channel)
        self.coupling = AffineCouplingCond(in_channel, cond_dim, affine=affine)

    def forward(self, x, cond):
        y, logdet = self.actnorm(x)
        y, det1 = self.invconv(y)
        y, det2 = self.coupling(y, cond)
        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2
        return y, logdet

    def reverse(self, y, cond):
        x = self.coupling.reverse(y, cond)
        x = self.invconv.reverse(x)
        x = self.actnorm.reverse(x)
        return x


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps

class BlockCond(nn.Module):
    def __init__(self, in_channel, n_flow, cond_dim, split=True, affine=True):
        super().__init__()
        squeeze_dim = in_channel * 4
        self.flows = nn.ModuleList([FlowCond(squeeze_dim, cond_dim, affine=affine) for _ in range(n_flow)])
        self.split = split

        if split:
            self.prior = CondPriorNet(in_channel * 2, in_channel * 4, cond_dim)
        else:
            self.prior = CondPriorNet(in_channel * 4, in_channel * 8, cond_dim)

    def forward(self, x, cond):
        B, C, H, W = x.shape

        # squeeze
        s = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 1, 3, 5, 2, 4).contiguous()
        y = s.view(B, C * 4, H // 2, W // 2)

        logdet = torch.zeros(B, device=x.device)
        for flow in self.flows:
            y, det = flow(y, cond)
            logdet = logdet + det

        if self.split:
            y, z_new = y.chunk(2, 1)
            mean, log_sd = self.prior(y, cond).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd).view(B, -1).sum(1)
        else:
            zero = torch.zeros_like(y)
            mean, log_sd = self.prior(zero, cond).chunk(2, 1)
            log_p = gaussian_log_p(y, mean, log_sd).view(B, -1).sum(1)
            z_new = y

        return y, logdet, log_p, z_new

    def reverse(self, y, eps, cond, reconstruct=False):
        x = y

        if reconstruct:
            x = torch.cat([y, eps], 1) if self.split else eps
        else:
            if self.split:
                mean, log_sd = self.prior(x, cond).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                x = torch.cat([y, z], 1)
            else:
                zero = torch.zeros_like(x)
                mean, log_sd = self.prior(zero, cond).chunk(2, 1)
                x = gaussian_sample(eps, mean, log_sd)

        for flow in reversed(self.flows):
            x = flow.reverse(x, cond)

        # unsqueeze
        B, C, H, W = x.shape
        u = x.view(B, C // 4, 2, 2, H, W).permute(0, 1, 4, 2, 5, 3).contiguous()
        u = u.view(B, C // 4, H * 2, W * 2)
        return u


class GlowCond(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, cond_dim=256, affine=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        n_channel = in_channel

        for _ in range(n_block - 1):
            self.blocks.append(BlockCond(n_channel, n_flow, cond_dim, split=True, affine=affine))
            n_channel *= 2

        self.blocks.append(BlockCond(n_channel, n_flow, cond_dim, split=False, affine=affine))

    def forward(self, x, cond):
        B = x.size(0)
        log_p_sum = torch.zeros(B, device=x.device)
        logdet = torch.zeros(B, device=x.device)

        out = x
        z_outs = []
        for block in self.blocks:
            out, det, log_p, z_new = block(out, cond)
            z_outs.append(z_new)
            logdet = logdet + det
            log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, eps_list, cond, reconstruct=False):
        x = None
        for i, block in enumerate(reversed(self.blocks)):
            if i == 0:
                x = block.reverse(eps_list[-1], eps_list[-1], cond, reconstruct=reconstruct)
            else:
                x = block.reverse(x, eps_list[-(i + 1)], cond, reconstruct=reconstruct)
        return x
