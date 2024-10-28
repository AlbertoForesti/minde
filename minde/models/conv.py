import torch
from torch import nn
from minde.models.mlp import exists, default
from functools import partial

class Conv2DBlock(nn.Module):

    def __init__(self, dim, dim_out, groups=8, shift_scale=True):
        super().__init__()

        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.act = nn.SiLU()
        # self.act = nn.Relu()
        self.norm = nn.GroupNorm(groups, dim)
        # self.norm = nn.BatchNorm1d( dim)
        self.shift_scale = shift_scale

    def forward(self, x, t=None):
        x = self.norm(x)
        x = self.act(x)
        x = self.proj(x)

        if exists(t):
            if self.shift_scale:
                scale, shift = t
                x = x * (scale.squeeze() + 1) + shift.squeeze()
            else:
                try:
                    x = x + t[...,None,None]
                except:
                    raise ValueError(f"Size mismatch. x size is {x.size()} and t size is {t.size()}")

        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=32, shift_scale=False):
        super().__init__()
        self.shift_scale = shift_scale
        self.mlp = nn.Sequential(
            nn.SiLU(),
            # nn.Linear(time_emb_dim, dim_out * 2)
            nn.Linear(time_emb_dim, dim_out*2 if shift_scale else dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Conv2DBlock(dim, dim_out, groups=groups,
                            shift_scale=shift_scale)
        self.block2 = Conv2DBlock(dim_out, dim_out, groups=groups,
                            shift_scale=shift_scale)
        # self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        """self.lin_layer = nn.Linear(
            dim, dim_out) if dim != dim_out else nn.Identity()"""
        
        if dim != dim_out:
            stride = 1
            self.projection = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(dim_out),
                nn.SiLU()
            )
        else:
            self.projection = nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):

            time_emb = self.mlp(time_emb)
            scale_shift = time_emb

        h = self.block1(x, t=scale_shift)

        h = self.block2(h)

        x = self.projection(x)

        try:
            return h + x
        except:
            raise ValueError(f"Size mismatch. h size is {h.size()} and x size is {x.size()}")

class UnetConv2D_simple(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=128,
        dim_mults=(1, 1),
        resnet_block_groups=8,
        time_dim=128,
        nb_var=1,
    ):
        super().__init__()

        # determine dimensions
        self.nb_var = nb_var
        init_dim = default(init_dim, dim)
        if init_dim == None:
            init_dim = dim * dim_mults[0]

        dim_in = dim
        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        self.init_lin = nn.Conv2d(dim, init_dim, 1)

        self.time_mlp = nn.Sequential(
            nn.Linear(nb_var, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            module = nn.ModuleList([block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                                    #        block_klass(dim_in, dim_in, time_emb_dim = time_dim)
                                    ])

            # module.append( Downsample(dim_in, dim_out) if not is_last else nn.Linear(dim_in, dim_out))
            self.downs.append(module)

        mid_dim = dims[-1]
        joint_dim = mid_dim
       # joint_dim = 24
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # self.mid_block2 = block_klass(joint_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            module = nn.ModuleList([block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                                    #       block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim)
                                    ])
            # module.append( Upsample(dim_out, dim_in) if not is_last else  nn.Linear(dim_out, dim_in))
            self.ups.append(module)

        # default_out_dim = channels * (1 if not learned_variance else 2)

        self.out_dim = dim_in

        self.final_res_block = block_klass(
            init_dim * 2, init_dim, time_emb_dim=time_dim)

        self.final_lin = Conv2DBlock(init_dim, dim, groups=8)

    def forward(self, x, t=None, std=None):
        t = t.reshape(t.size(0), self.nb_var)

        if len(x.shape) == 3:
            x = x.unsqueeze(1) # Add channel dimension for greyscale images
        elif len(x.shape) != 4:
            raise ValueError("Input tensor must be 3D or 4D")

        x = self.init_lin(x.float())

        r = x.clone()

        t = self.time_mlp(t).squeeze()

        h = []

        for blocks in self.downs:

            block1 = blocks[0]

            x = block1(x, t)

            h.append(x)
       #     x = downsample(x)

        # x = self.mid_block1(x, t)

        # x = self.mid_block2(x, t)

        for blocks in self.ups:

            block1 = blocks[0]
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            # x = torch.cat((x, h.pop()), dim = 1)
            # x = block2(x, t)

           # x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)

        if std != None:
            return self.final_lin(x) / std
        else:
            return self.final_lin(x)