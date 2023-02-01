import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from commons import clip_grad_value_
import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, DistributedBucketSampler
from models.synthesizer import SynthesizerTrn
from models.discriminator import MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

import pytorch_lightning as pl


class VitsDataModule(pl.LightningDataModule):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps

    def prepare_data(self):
        return

    def setup(self, stage):
        self.train = TextAudioLoader(self.hps.dataset, train=True)
        self.val = TextAudioLoader(self.hps.dataset, train=False)

    def train_dataloader(self):
        train_sampler = DistributedBucketSampler(
            self.train,
            hps.train.batch_size,
            [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        )
        return DataLoader(
            self.train,
            num_workers=4,
            pin_memory=True,
            collate_fn=TextAudioCollate(),
            persistent_workers=True,
            batch_sampler=train_sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            num_workers=4,
            batch_size=hps.train.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=TextAudioCollate(),
            persistent_workers=True,
        )


class Vits(pl.LightningModule):
    def __init__(self, hps):
        super().__init__()
        self.save_hyperparameters()
        self.hps = hps
        self.lr = hps.train.learning_rate

        self.generator = SynthesizerTrn(
            len(symbols),
            hps.dataset.filter_length // 2 + 1,
            hps.train.segment_size // hps.dataset.hop_length,
            **hps.model
        )
        self.discriminator = MultiPeriodDiscriminator(hps.model.use_spectral_norm)

        self.automatic_optimization = False

    def forward(self, x, x_lengths, spec, spec_lengths):
        return self.generator(x, x_lengths, spec, spec_lengths)

    def training_step(self, batch, batch_idx):
        (optim_generator, optim_discriminator) = self.optimizers()
        (x, x_lengths, spec, spec_lengths, y, _) = batch

        # Generate
        (
            y_hat,
            l_length,
            attn,
            ids_slice,
            x_mask,
            z_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
        ) = self(x, x_lengths, spec, spec_lengths)
        mel = self.spec_to_mel_torch(spec)
        y_mel = commons.slice_segments(
            mel,
            ids_slice,
            self.hps.train.segment_size // self.hps.dataset.hop_length,
        )
        y_hat_mel = self.mel_spectrogram_torch(y_hat)
        y = commons.slice_segments(
            y, ids_slice * self.hps.dataset.hop_length, self.hps.train.segment_size
        )

        y_d_hat_r, y_d_hat_g, _, _ = self.discriminator(y, y_hat.detach())
        with autocast(enabled=False):
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )
            loss_disc_all = loss_disc

        optim_discriminator.zero_grad(set_to_none=True)
        self.manual_backward(loss_disc_all)
        discriminator_grad_clip = clip_grad_value_(
            self.discriminator.parameters(), None
        )
        optim_discriminator.step()

        self.log(
            "train/discriminator/grad_clip",
            discriminator_grad_clip,
            on_step=False,
            on_epoch=True,
        )
        self.log("train/discriminator/all", loss_disc, on_step=False, on_epoch=True)

        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(y, y_hat)
        with autocast(enabled=False):
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hps.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hps.train.c_kl

            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
        optim_generator.zero_grad(set_to_none=True)
        self.manual_backward(loss_gen_all)
        generator_grad_clip = clip_grad_value_(self.generator.parameters(), None)
        optim_generator.step()

        self.log("train/generator/duration", loss_dur, on_step=False, on_epoch=True)
        self.log("train/generator/mel", loss_mel, on_step=False, on_epoch=True)
        self.log("train/generator/kl", loss_kl, on_step=False, on_epoch=True)
        self.log("train/generator/feature", loss_fm, on_step=False, on_epoch=True)
        self.log("train/generator/generator", loss_gen, on_step=False, on_epoch=True)
        self.log("train/generator/all", loss_gen_all, on_step=False, on_epoch=True)
        self.log(
            "train/generator/grad_clip",
            generator_grad_clip,
            on_step=False,
            on_epoch=True,
        )

    def validation_step(self, batch, batch_idx):
        (x, x_lengths, spec, spec_lengths, y, y_lengths) = batch

        # remove else
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]

        y_hat, attn, mask, *_ = self.generator.infer(x, x_lengths, max_len=1000)
        y_hat_lengths = mask.sum([1, 2]).long() * self.hps.dataset.hop_length

        mel = spec_to_mel_torch(
            spec,
            self.hps.dataset.filter_length,
            self.hps.dataset.n_mel_channels,
            self.hps.dataset.sampling_rate,
            self.hps.dataset.mel_fmin,
            self.hps.dataset.mel_fmax,
        )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            self.hps.dataset.filter_length,
            self.hps.dataset.n_mel_channels,
            self.hps.dataset.sampling_rate,
            self.hps.dataset.hop_length,
            self.hps.dataset.win_length,
            self.hps.dataset.mel_fmin,
            self.hps.dataset.mel_fmax,
        )

        self.logger.experiment.add_image(
            "eval_figures/alignment",
            utils.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy()),
            self.global_step,
            dataformats="HWC",
        )
        self.logger.experiment.add_image(
            "eval_figures/fake/mel",
            utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
            self.global_step,
            dataformats="HWC",
        )
        self.logger.experiment.add_image(
            "eval_figures/real/mel",
            utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy()),
            self.global_step,
            dataformats="HWC",
        )
        self.logger.experiment.add_audio(
            "eval_figures/fake/audio",
            y_hat[0, :, : y_hat_lengths[0]],
            self.global_step,
            self.hps.dataset.sampling_rate,
        )
        self.logger.experiment.add_audio(
            "eval_figures/real/audio",
            y[0, :, : y_lengths[0]],
            self.global_step,
            self.hps.dataset.sampling_rate,
        )

    def configure_optimizers(self):
        optim_generator = torch.optim.AdamW(
            self.generator.parameters(),
            self.lr,
            betas=self.hps.train.betas,
            eps=self.hps.train.eps,
        )
        optim_discriminator = torch.optim.AdamW(
            self.discriminator.parameters(),
            self.lr,
            betas=self.hps.train.betas,
            eps=self.hps.train.eps,
        )
        scheduler_generator = torch.optim.lr_scheduler.ExponentialLR(
            optim_generator, gamma=self.hps.train.lr_decay
        )
        scheduler_discriminator = torch.optim.lr_scheduler.ExponentialLR(
            optim_discriminator, gamma=self.hps.train.lr_decay
        )
        return [optim_generator, optim_discriminator], [
            scheduler_generator,
            scheduler_discriminator,
        ]
        # optim_generator = VeLO(self.generator.parameters(), 100000, device='cpu')
        # optim_discriminator = VeLO(self.discriminator.parameters(), 100000, device='cpu')
        # return [optim_generator, optim_discriminator]

    def spec_to_mel_torch(self, spec):
        return spec_to_mel_torch(
            spec,
            self.hps.dataset.filter_length,
            self.hps.dataset.n_mel_channels,
            self.hps.dataset.sampling_rate,
            self.hps.dataset.mel_fmin,
            self.hps.dataset.mel_fmax,
        )

    def mel_spectrogram_torch(self, y_hat):
        return mel_spectrogram_torch(
            y_hat.squeeze(1),
            self.hps.dataset.filter_length,
            self.hps.dataset.n_mel_channels,
            self.hps.dataset.sampling_rate,
            self.hps.dataset.hop_length,
            self.hps.dataset.win_length,
            self.hps.dataset.mel_fmin,
            self.hps.dataset.mel_fmax,
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    hps = utils.get_hparams()
    torch.manual_seed(hps.train.seed)

    model = Vits(hps)
    data = VitsDataModule(hps)

    trainer = pl.Trainer(accelerator="gpu", devices=1, precision=16, max_steps=500000)
    # trainer.tune(model, data)
    trainer.fit(model, data)
