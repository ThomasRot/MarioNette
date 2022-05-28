"""Model training interface."""
import torch as th
from torch.distributions import Beta
from torch.nn import functional as F
import ttools
from ttools.modules import image_operators as imops

LOG = ttools.get_logger(__name__)


class Interface(ttools.ModelInterface):
    def __init__(self, model, device="cpu", lr=1e-4, w_beta=0, w_probs=0,
                 lr_bg=None, background=None):
        self.model = model

        if lr_bg is None:
            lr_bg = lr

        self.opt = th.optim.AdamW(model.parameters(), lr=lr)

        self.device = device
        self.model.to(device)
        self.background = background
        if background is not None:
            self.background.to(device)
            self.opt_bg = th.optim.AdamW(
                self.background.parameters(), lr=lr_bg)
        else:
            self.opt_bg = None

        self.w_beta = w_beta
        self.w_probs = w_probs

        self.beta = Beta(th.tensor(2.).to(device), th.tensor(2.).to(device))

        self.loss = th.nn.MSELoss()
        self.loss.to(device)

    def forward(self, im, hard=False):
        if self.background is not None:
            bg, _ = self.background()
        else:
            bg = None
        return self.model(im, bg, hard=hard)

    def training_step(self, batch):
        im = batch["im"].to(self.device)

        fwd_data = self.forward(im)
        fwd_data_hard = self.forward(im, hard=True)

        out = fwd_data["reconstruction"]
        layers = fwd_data["layers"]
        out_hard = fwd_data_hard["reconstruction"]
        layers_hard = fwd_data_hard["layers"]
        im = imops.crop_like(im, out)

        learned_dict = fwd_data["dict"]
        dict_codes = fwd_data["dict_codes"]
        # [B * NL, LZ, LZ, dim_z]
        im_codes = fwd_data["im_codes"]
        weights = fwd_data["weights"]
        # [B, NL, LZ * LZ, dim_z]
        probs = fwd_data["probs"]
        B, NL = probs.shape[:2]
        LZ, dim_z = im_codes.shape[2:]
        # [B * NL * LZ * LZ, 2]
        shifts = fwd_data["shifts"]

        print()

        rec_loss = self.loss(out, im)
        beta_loss = (self.beta.log_prob(
            weights.clamp(1e-5, 1 - 1e-5)).exp().mean() + self.beta.log_prob(
            probs.clamp(1e-5, 1 - 1e-5)).exp().mean()) / 2

        probs_loss = probs.abs()
        object_consistency_loss = self.object_consistency(im_codes.reshape(B * NL, LZ, LZ, dim_z),
                                                          probs.reshape(B * NL, LZ, LZ),
                                                          shifts.reshape(B * NL, LZ, LZ, 2))
        self.opt.zero_grad()
        if self.opt_bg is not None:
            self.opt_bg.zero_grad()

        w_probs = th.tensor(self.w_probs).to(probs_loss)[None, :, None] \
            .expand_as(probs_loss)
        loss = rec_loss + self.w_beta * beta_loss + \
               (w_probs * probs_loss).mean() + object_consistency_loss

        # print(f'{learned_dict.shape=}, {dict_codes.shape=}, {im_codes.shape=}, {weights.shape=}, {probs.shape=}')

        loss.backward()

        self.opt.step()
        if self.opt_bg is not None:
            self.opt_bg.step()

        with th.no_grad():
            psnr = -10 * th.log10(F.mse_loss(out, im))
            psnr_hard = -10 * th.log10(F.mse_loss(out_hard, im))

        return {
            "rec_loss": rec_loss.item(),
            "beta_loss": beta_loss.item(),
            "psnr": psnr.item(),
            "psnr_hard": psnr_hard.item(),
            "out": out.detach(),
            "layers": layers.detach(),
            "out_hard": out_hard.detach(),
            "layers_hard": layers_hard.detach(),
            "dict": learned_dict.detach(),
            "probs_loss": probs_loss.mean().item(),
            "im_codes": im_codes.detach(),
            "dict_codes": dict_codes.detach(),
            "background": fwd_data["background"].detach()
        }

    def object_consistency(self, what, pres, shift, T=4):
        BT, NL, LZ, LZ, z_dim = what.shape
        z_whats = what.reshape(-1, T, NL, LZ, LZ, z_dim).transpose(0, 1).reshape(-1, T, LZ, LZ, z_dim)
        B = what.shape[0]
        z_where = shift.reshape(B, T, NL, LZ, LZ, 2).transpose(0, 1).reshape(T, B * NL, LZ, LZ, 2)
        z_pres = pres.reshape(B, T, NL, LZ, LZ).transpose(0, 1).reshape(T, B * NL, LZ, LZ)
        z_pres_idx = (z_pres[:-1] > 0.5).nonzero(as_tuple=False)

        # (T, B*NZ, G+2, G+2)
        z_pres_same_padding = torch.nn.functional.pad(z_pres, (1, 1, 1, 1), mode='replicate')
        # (T, B*NZ, G+2, G+2, D)
        z_what_same_padding = torch.nn.functional.pad(z_whats, (0, 0, 1, 1, 1, 1), mode='replicate')
        # (T, B*NZ, G+2, G+2, 2)
        z_where_same_padding = torch.nn.functional.pad(z_where, (0, 0, 1, 1, 1, 1), mode='replicate')
        # idx: (4,)
        object_consistency_loss = torch.tensor(0.0).to(z_whats.device)
        for idx in z_pres_idx:
            # (3, 3)
            z_pres_area = z_pres_same_padding[idx[0] + 1, idx[1], idx[2]:idx[2] + 3, idx[3]:idx[3] + 3]
            # (3, 3, D)
            z_what_area = z_what_same_padding[idx[0] + 1, idx[1], idx[2]:idx[2] + 3, idx[3]:idx[3] + 3]
            # (3, 3, 4)
            z_where_area = z_where_same_padding[idx[0] + 1, idx[1], idx[2]:idx[2] + 3, idx[3]:idx[3] + 3]
            # Tuple((#hits,) (#hits,))
            z_what_idx = (z_pres_area > arch.object_threshold).nonzero(as_tuple=True)
            # (1, D)
            z_what_prior = z_whats[idx.tensor_split(4)]
            # (1, 4)
            z_where_prior = z_where[idx.tensor_split(4)]
            # (#hits, D)
            z_whats_now = z_what_area[z_what_idx]
            # (#hits, 4)
            z_where_now = z_where_area[z_what_idx]
            if z_whats_now.nelement() == 0:
                continue
            # (#hits,)
            z_sim = cos(z_what_prior, z_whats_now)
            if z_whats_now.shape[0] > 1:
                similarity_max_idx = z_sim.argmax()
                pos_dif_min = nn.functional.mse_loss(z_where_prior.expand_as(z_where_now), z_where_now,
                                                     reduction='none').sum(dim=-1).argmin()
                if pos_dif_min != similarity_max_idx:
                    continue
                object_consistency_loss += -5.0 * z_sim[similarity_max_idx] + torch.sum(z_sim)
            else:
                object_consistency_loss += -5.0 * torch.max(z_sim) + torch.sum(z_sim)
        return object_consistency_loss, torch.tensor(len(z_pres_idx)).to(z_whats.device)
