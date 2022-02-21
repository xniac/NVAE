import torch

from model import *

class ConditionalAutoEncoder(nn.Module):
    def __init__(self, args, writer, arch_instance):
        super(ConditionalAutoEncoder, self).__init__()
        self.writer = writer
        self.arch_instance = arch_instance
        self.dataset = args.dataset
        self.crop_output = self.dataset in {'mnist', 'omniglot', 'stacked_mnist'}
        self.use_se = args.use_se
        self.res_dist = args.res_dist
        self.num_bits = args.num_x_bits
        self.u_shape = args.u_shape

        self.num_latent_scales = args.num_latent_scales         # number of spatial scales that latent layers will reside
        self.num_groups_per_scale = args.num_groups_per_scale   # number of groups of latent vars. per scale
        self.num_latent_per_group = args.num_latent_per_group   # number of latent vars. per group
        self.groups_per_scale = groups_per_scale(self.num_latent_scales, self.num_groups_per_scale, args.ada_groups,
                                                 minimum_groups=args.min_groups_per_scale)

        self.vanilla_vae = self.num_latent_scales == 1 and self.num_groups_per_scale == 1

        # encoder parameteres
        self.num_channels_enc = args.num_channels_enc
        self.num_channels_dec = args.num_channels_dec
        self.num_preprocess_blocks = args.num_preprocess_blocks  # block is defined as series of Normal followed by Down
        self.num_preprocess_cells = args.num_preprocess_cells   # number of cells per block
        self.num_cell_per_cond_enc = args.num_cell_per_cond_enc  # number of cell for each conditional in encoder

        # decoder parameters
        # self.num_channels_dec = args.num_channels_dec
        self.num_postprocess_blocks = args.num_postprocess_blocks
        self.num_postprocess_cells = args.num_postprocess_cells
        self.num_cell_per_cond_dec = args.num_cell_per_cond_dec  # number of cell for each conditional in decoder

        # general cell parameters
        self.input_size = get_input_size(self.dataset)

        # decoder param
        self.num_mix_output = args.num_mixture_dec

        # used for generative purpose
        c_scaling = CHANNEL_MULT ** (self.num_preprocess_blocks + self.num_latent_scales - 1)
        spatial_scaling = 2 ** (self.num_preprocess_blocks + self.num_latent_scales - 1)
        prior_ftr0_size = (int(c_scaling * self.num_channels_dec), self.input_size // spatial_scaling,
                           self.input_size // spatial_scaling)
        self.prior_ftr0 = nn.Parameter(torch.rand(size=prior_ftr0_size), requires_grad=True)
        self.z0_size = [self.num_latent_per_group, self.input_size // spatial_scaling, self.input_size // spatial_scaling]

        self.stem, self.cond_stem = self.init_stem()
        self.pre_process, mult = self.init_pre_process(mult=1)

        if self.vanilla_vae:
            self.enc_tower = []
            self.cond_enc_tower = []
        else:
            self.enc_tower, self.cond_enc_tower, mult = self.init_encoder_tower(mult)

        self.with_nf = args.num_nf > 0
        self.num_flows = args.num_nf

        self.enc0 = self.init_encoder0(mult)
        self.enc_sampler, self.dec_sampler, self.nf_cells, self.enc_kv, self.dec_kv, self.query = \
            self.init_normal_sampler(mult)

        if self.vanilla_vae:
            self.dec_tower = []
            self.stem_decoder = Conv2D(self.num_latent_per_group, mult * self.num_channels_enc, (1, 1), bias=True)
        else:
            self.dec_tower, mult = self.init_decoder_tower(mult)

        self.post_process, mult = self.init_post_process(mult)

        self.image_conditional = self.init_image_conditional(mult)

        # collect all norm params in Conv2D and gamma param in batchnorm
        self.all_log_norm = []
        self.all_conv_layers = []
        self.all_bn_layers = []
        for n, layer in self.named_modules():
            # if isinstance(layer, Conv2D) and '_ops' in n:   # only chose those in cell
            if isinstance(layer, Conv2D) or isinstance(layer, ARConv2d):
                self.all_log_norm.append(layer.log_weight_norm)
                self.all_conv_layers.append(layer)
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.SyncBatchNorm) or \
                    isinstance(layer, SyncBatchNormSwish):
                self.all_bn_layers.append(layer)

        print('len log norm:', len(self.all_log_norm))
        print('len bn:', len(self.all_bn_layers))
        # left/right singular vectors used for SR
        self.sr_u = {}
        self.sr_v = {}
        self.num_power_iter = 4

    def init_stem(self):
        Cout = self.num_channels_enc
        Cin = 1 if self.dataset in {'mnist', 'omniglot'} else 3
        stem = Conv2D(Cin*2, Cout, 3, padding=1, bias=True)
        cond_stem = Conv2D(Cin, Cout, 3, padding=1, bias=True)
        return stem, cond_stem

    # def init_cond_stem(self):
    #     Cout = self.num_channels_enc
    #     Cin = 1 if self.dataset in {'mnist', 'omniglot'} else 3
    #     cond_stem = Conv2D(Cin, Cout, 3, padding=1, bias=True)
    #     return cond_stem

    def init_pre_process(self, mult):
        pre_process = nn.ModuleList()
        for b in range(self.num_preprocess_blocks):
            for c in range(self.num_preprocess_cells):
                if c == self.num_preprocess_cells - 1:
                    arch = self.arch_instance['down_pre']
                    num_ci = int(self.num_channels_enc * mult)
                    num_co = int(CHANNEL_MULT * num_ci)
                    cell = Cell(num_ci, num_co, cell_type='down_pre', arch=arch, use_se=self.use_se)
                    mult = CHANNEL_MULT * mult
                else:
                    arch = self.arch_instance['normal_pre']
                    num_c = self.num_channels_enc * mult
                    cell = Cell(num_c, num_c, cell_type='normal_pre', arch=arch, use_se=self.use_se)

                pre_process.append(cell)

        return pre_process, mult

    def init_encoder_tower(self, mult):
        enc_tower = nn.ModuleList()
        cond_enc_tower = nn.ModuleList()
        for s in range(self.num_latent_scales):
            for g in range(self.groups_per_scale[s]):
                for c in range(self.num_cell_per_cond_enc):
                    arch = self.arch_instance['normal_enc']
                    num_c = int(self.num_channels_enc * mult)
                    cell = Cell(num_c, num_c, cell_type='normal_enc', arch=arch, use_se=self.use_se)
                    cond_cell = Cell(num_c, num_c, cell_type='normal_enc', arch=arch, use_se=self.use_se)
                    enc_tower.append(cell)
                    cond_enc_tower.append(cond_cell)

                # add encoder combiner
                if not (s == self.num_latent_scales - 1 and g == self.groups_per_scale[s] - 1):
                    num_ce = int(self.num_channels_enc * mult)
                    num_cd = int(self.num_channels_dec * mult)
                    cell = EncCombinerCell(num_ce, num_cd, num_ce, cell_type='combiner_enc')
                    cond_cell = EncCombinerCell(num_ce, num_cd, num_ce, cell_type='combiner_enc')
                    enc_tower.append(cell)
                    cond_enc_tower.append(cond_cell)

            # down cells after finishing a scale
            if s < self.num_latent_scales - 1:
                arch = self.arch_instance['down_enc']
                num_ci = int(self.num_channels_enc * mult)
                num_co = int(CHANNEL_MULT * num_ci)
                cell = Cell(num_ci, num_co, cell_type='down_enc', arch=arch, use_se=self.use_se)
                cond_cell = Cell(num_ci, num_co, cell_type='down_enc', arch=arch, use_se=self.use_se)
                enc_tower.append(cell)
                cond_enc_tower.append(cond_cell)
                mult = CHANNEL_MULT * mult

        return enc_tower, cond_enc_tower, mult

    def init_encoder0(self, mult):
        num_c = int(self.num_channels_enc * mult)
        cell = nn.Sequential(
            nn.ELU(),
            Conv2D(num_c, num_c, kernel_size=1, bias=True),
            nn.ELU())
        return cell

    def init_normal_sampler(self, mult):
        enc_sampler, dec_sampler, nf_cells = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        enc_kv, dec_kv, query = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for s in range(self.num_latent_scales):
            for g in range(self.groups_per_scale[self.num_latent_scales - s - 1]):
                # build mu, sigma generator for encoder
                num_c = int(self.num_channels_enc * mult)
                cell = Conv2D(num_c, 2 * self.num_latent_per_group, kernel_size=3, padding=1, bias=True)
                enc_sampler.append(cell)
                # build NF
                # for n in range(self.num_flows):
                #     arch = self.arch_instance['ar_nn']
                #     num_c1 = int(self.num_channels_enc * mult)
                #     num_c2 = 8 * self.num_latent_per_group  # use 8x features
                #     nf_cells.append(PairedCellAR(self.num_latent_per_group, num_c1, num_c2, arch))
                if not (s == 0 and g == 0):  # for the first group, we use a fixed standard Normal.
                    num_c = int(self.num_channels_dec * mult)
                    if self.u_shape:
                        num_c *= 2
                    cell = nn.Sequential(
                        nn.ELU(),
                        Conv2D(num_c, 2 * self.num_latent_per_group, kernel_size=1, padding=0, bias=True))
                    dec_sampler.append(cell)
                else:
                    num_c = int(self.num_channels_dec * mult)
                    cell = nn.Sequential(
                        nn.ELU(),
                        Conv2D(num_c, 2 * self.num_latent_per_group, kernel_size=1, padding=0, bias=True))
                    dec_sampler.append(cell)

            mult = mult / CHANNEL_MULT

        return enc_sampler, dec_sampler, nf_cells, enc_kv, dec_kv, query

    def init_decoder_tower(self, mult):
        # create decoder tower
        dec_tower = nn.ModuleList()
        for s in range(self.num_latent_scales):
            for g in range(self.groups_per_scale[self.num_latent_scales - s - 1]):
                num_c = int(self.num_channels_dec * mult)
                if not (s == 0 and g == 0):
                    for c in range(self.num_cell_per_cond_dec):
                        arch = self.arch_instance['normal_dec']
                        cell = Cell(num_c, num_c, cell_type='normal_dec', arch=arch, use_se=self.use_se)
                        dec_tower.append(cell)

                cell = DecCombinerCell(num_c, self.num_latent_per_group, num_c, cell_type='combiner_dec')
                dec_tower.append(cell)

            # down cells after finishing a scale
            if s < self.num_latent_scales - 1:
                arch = self.arch_instance['up_dec']
                num_ci = int(self.num_channels_dec * mult)
                num_co = int(num_ci / CHANNEL_MULT)
                cell = Cell(num_ci, num_co, cell_type='up_dec', arch=arch, use_se=self.use_se)
                dec_tower.append(cell)
                mult = mult / CHANNEL_MULT

        return dec_tower, mult

    def init_post_process(self, mult):
        post_process = nn.ModuleList()
        for b in range(self.num_postprocess_blocks):
            for c in range(self.num_postprocess_cells):
                if c == 0:
                    arch = self.arch_instance['up_post']
                    num_ci = int(self.num_channels_dec * mult)
                    num_co = int(num_ci / CHANNEL_MULT)
                    cell = Cell(num_ci, num_co, cell_type='up_post', arch=arch, use_se=self.use_se)
                    mult = mult / CHANNEL_MULT
                else:
                    arch = self.arch_instance['normal_post']
                    num_c = int(self.num_channels_dec * mult)
                    cell = Cell(num_c, num_c, cell_type='normal_post', arch=arch, use_se=self.use_se)

                post_process.append(cell)

        return post_process, mult

    def init_image_conditional(self, mult):
        C_in = int(self.num_channels_dec * mult)
        if self.dataset in {'mnist', 'omniglot'}:
            C_out = 1
        else:
            if self.num_mix_output == 1:
                C_out = 2 * 3
            else:
                C_out = 10 * self.num_mix_output
        return nn.Sequential(nn.ELU(),
                             Conv2D(C_in, C_out, 3, padding=1, bias=True))

    def forward(self, x, cond_x):
        # concat the x with condition_x at the channel dim, passing through the input together
        x_stack = torch.cat([x, cond_x], dim=1)
        s = self.stem(2 * x_stack - 1.0)
        cond_s = self.cond_stem(2 * cond_x - 1)

        # perform pre-processing
        for cell in self.pre_process:
            s = cell(s)
            cond_s = cell(cond_s)

        # run the main encoder tower
        combiner_cells_enc = []
        combiner_cells_s = []
        for cell in self.enc_tower:
            # combiner_cell_s record left part of q(z_i|x), z_l, z_l-1, ..., z_1
            if cell.cell_type == 'combiner_enc':
                combiner_cells_enc.append(cell)
                combiner_cells_s.append(s)
            else:
                s = cell(s)
        combiner_cells_enc.reverse()
        combiner_cells_s.reverse()

        if self.u_shape:
            combiner_cells_cond_enc = []
            combiner_cells_cond_s = []
            for cell in self.cond_enc_tower:
                if cell.cell_type == 'combiner_enc':
                    combiner_cells_cond_enc.append(cell)
                    combiner_cells_cond_s.append(cond_s)
                else:
                    cond_s = cell(cond_s)
            combiner_cells_cond_enc.reverse()
            combiner_cells_cond_s.reverse()

        # reverse combiner cells and their input for decoder: left part of q(z_i|x), z_1, ..., z_l-1, z_l


        idx_dec = 0

        # prior for z0
        # TODO: still use the original encoder and sample

        cond_ftr = self.enc0(cond_s)
        cond_param = self.dec_sampler[idx_dec](cond_ftr)
        mu_p, log_sig_p = torch.chunk(cond_param, 2, dim=1)

        ftr = self.enc0(s)                            # this reduces the channel dimension
        param0 = self.enc_sampler[idx_dec](ftr)
        mu_q, log_sig_q = torch.chunk(param0, 2, dim=1)
        dist = Normal(mu_q+mu_p, log_sig_q+log_sig_p)   # for the first approx. posterior
        z, _ = dist.sample()
        log_q_conv = dist.log_p(z)

        # apply normalizing flows

        # all_q records q(z_l|x, z_<l)
        all_q = [dist]
        all_log_q = [log_q_conv]
        dist = Normal(mu_p, log_sig_p)
        # z, _ = dist.sample() #TODO: whether z need to be resampled?
        log_p_conv = dist.log_p(z)
        # all_p records p(z_l|z<l)
        all_p = [dist]
        all_log_p = [log_p_conv]

        # To make sure we do not pass any deterministic features from x to decoder.
        s = 0
        s = cond_s
        for cell in self.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    # form prior
                    if self.u_shape:
                        cond_ftr = combiner_cells_cond_enc[idx_dec - 1](combiner_cells_cond_s[idx_dec - 1], s)
                        stack_s = torch.cat([cond_ftr, s], dim=1)
                        param = self.dec_sampler[idx_dec](stack_s)
                    else:
                        param = self.dec_sampler[idx_dec](s)

                    mu_p, log_sig_p = torch.chunk(param, 2, dim=1)

                    # form encoder
                    ftr = combiner_cells_enc[idx_dec - 1](combiner_cells_s[idx_dec - 1], s)
                    param = self.enc_sampler[idx_dec](ftr)
                    mu_q, log_sig_q = torch.chunk(param, 2, dim=1)
                    dist = Normal(mu_p + mu_q, log_sig_p + log_sig_q) if self.res_dist else Normal(mu_q, log_sig_q)
                    z, _ = dist.sample()
                    log_q_conv = dist.log_p(z)
                    # apply NF

                    all_log_q.append(log_q_conv)
                    all_q.append(dist)

                    # evaluate log_p(z)
                    dist = Normal(mu_p, log_sig_p)
                    log_p_conv = dist.log_p(z)
                    all_p.append(dist)
                    all_log_p.append(log_p_conv)

                # 'combiner_dec'
                s = cell(s, z)
                idx_dec += 1
            else:
                s = cell(s)

        if self.vanilla_vae:
            s = self.stem_decoder(z)

        for cell in self.post_process:
            s = cell(s)

        logits = self.image_conditional(s)

        # compute kl
        kl_all = []
        kl_diag = []
        log_p, log_q = 0., 0.
        for q, p, log_q_conv, log_p_conv in zip(all_q, all_p, all_log_q, all_log_p):

            kl_per_var = q.kl(p)

            kl_diag.append(torch.mean(torch.sum(kl_per_var, dim=[2, 3]), dim=0))
            kl_all.append(torch.sum(kl_per_var, dim=[1, 2, 3]))
            log_q += torch.sum(log_q_conv, dim=[1, 2, 3])
            log_p += torch.sum(log_p_conv, dim=[1, 2, 3])

        return logits, log_q, log_p, kl_all, kl_diag

    def sample(self, cond_x):

        cond_s = self.cond_stem(2 * cond_x - 1)
        for cell in self.pre_process:
            cond_s = cell(cond_s)

        if self.u_shape:
            combiner_cells_cond_enc = []
            combiner_cells_cond_s = []
            for cell in self.cond_enc_tower:
                if cell.cell_type == 'combiner_enc':
                    combiner_cells_cond_enc.append(cell)
                    combiner_cells_cond_s.append(cond_s)
                else:
                    cond_s = cell(cond_s)
            combiner_cells_cond_enc.reverse()
            combiner_cells_cond_s.reverse()

        scale_ind = 0
        idx_dec = 0

        # prior for z0
        # TODO: still use the original encoder and sample

        cond_ftr = self.enc0(cond_s)
        cond_param = self.dec_sampler[idx_dec](cond_ftr)
        mu_p, log_sig_p = torch.chunk(cond_param, 2, dim=1)
        dist = Normal(mu_p, log_sig_p)
        z, _ = dist.sample()

        # To make sure we do not pass any deterministic features from x to decoder.
        s = cond_s

        for cell in self.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    # form prior
                    if self.u_shape:
                        cond_ftr = combiner_cells_cond_enc[idx_dec - 1](combiner_cells_cond_s[idx_dec - 1], s)
                        stack_s = torch.cat([cond_ftr, s], dim=1)
                        param = self.dec_sampler[idx_dec](stack_s)
                    else:
                        param = self.dec_sampler[idx_dec](s)

                    mu_p, log_sig_p = torch.chunk(param, 2, dim=1)

                    # evaluate log_p(z)
                    dist = Normal(mu_p, log_sig_p)
                    z, _ = dist.sample()

                # 'combiner_dec'
                s = cell(s, z)
                idx_dec += 1
            else:
                s = cell(s)

        if self.vanilla_vae:
            s = self.stem_decoder(z)

        for cell in self.post_process:
            s = cell(s)

        logits = self.image_conditional(s)
        return logits

    def decoder_output(self, logits):
        if self.dataset in {'mnist', 'omniglot'}:
            return Bernoulli(logits=logits)
        elif self.dataset in {'stacked_mnist', 'cifar10', 'celeba_64', 'celeba_256', 'imagenet_32', 'imagenet_64', 'ffhq',
                              'lsun_bedroom_128', 'lsun_bedroom_256', 'lsun_church_64', 'lsun_church_128'}:
            if self.num_mix_output == 1:
                return NormalDecoder(logits, num_bits=self.num_bits)
            else:
                return DiscMixLogistic(logits, self.num_mix_output, num_bits=self.num_bits)
        else:
            raise NotImplementedError