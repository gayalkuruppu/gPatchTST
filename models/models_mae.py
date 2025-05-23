# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from .utils.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                 masking_method='random', mask_ratio=0.75):
        super().__init__()
        self.image_size = img_size # [f x t]
        self.patch_size = patch_size # [f x t]
        self.in_chans = in_chans
        self.masking_method = masking_method
        self.mask_ratio = mask_ratio

        self.num_patches_h = img_size[0] // patch_size[0]
        self.num_patches_w = img_size[1] // patch_size[1]

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, 
                #   qk_scale=None, 
                  norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
             Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, 
                #    qk_scale=None, 
                   norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.num_patches_h, self.num_patches_w), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], (self.num_patches_h, self.num_patches_w), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p1, p2 = self.patch_embed.patch_size
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        assert imgs.shape[2] % p1 == 0 and imgs.shape[3] % p2 == 0

        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, self.num_patches_h, p1, self.num_patches_w, p2))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], self.num_patches_h * self.num_patches_w, p1*p2*self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p1, p2 = self.patch_embed.patch_size
        assert self.num_patches_h * self.num_patches_w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], self.num_patches_h, self.num_patches_w, p1, p2, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, self.num_patches_h * p1, self.num_patches_w * p2))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    # def random_tf_masking(self, x, masking_ratio):
    #     """
    #     Perform random masking along frequency and time dimensions.
    #     x: [N, L, D], sequence
    #     freq_masking_ratio: Ratio of frequency patches to mask
    #     time_masking_ratio: Ratio of time patches to mask
    #     """
    #     freq_masking_ratio, time_masking_ratio = masking_ratio
    #     N, L, D = x.shape  # batch, length, dim

    #     # Calculate the number of patches along frequency and time dimensions
    #     f_patches = self.image_size[0] // self.patch_size[0]
    #     t_patches = self.image_size[1] // self.patch_size[1]

    #     # Number of rows and columns to mask
    #     freq_masked_patches = int(freq_masking_ratio * f_patches)
    #     time_masked_patches = int(time_masking_ratio * t_patches)

    #     # Randomly select rows and columns to mask
    #     freq_mask_indices = torch.randperm(f_patches, device=x.device)[:freq_masked_patches]
    #     time_mask_indices = torch.randperm(t_patches, device=x.device)[:time_masked_patches]

    #     # Convert to 1D indices (flattened from 2D patch grid)
    #     masked_rows = freq_mask_indices[:, None] * t_patches  # Shape (freq_masked_patches, 1)
    #     masked_rows = masked_rows + torch.arange(t_patches, device=x.device)  # Broadcasted
    #     masked_rows = masked_rows.flatten()

    #     masked_cols = torch.arange(f_patches, device=x.device)[:, None] * t_patches
    #     masked_cols = masked_cols + time_mask_indices  # Shape (f_patches, time_masked_patches)
    #     masked_cols = masked_cols.flatten()

    #     # Union of row and column masks
    #     masked_indices = torch.cat([masked_rows, masked_cols]).unique()
    #     all_indices = torch.arange(f_patches * t_patches, device=x.device)
    #     unmasked_indices = all_indices[~torch.isin(all_indices, masked_indices)]

    #     # Gather unmasked tokens
    #     x_unmasked = torch.gather(x, dim=1, index=unmasked_indices.unsqueeze(-1).repeat(1, 1, D))

    #     # Generate binary mask: 0 is keep, 1 is remove
    #     mask = torch.ones([N, L], device=x.device)
    #     mask[unmasked_indices] = 0

    #     return x_unmasked, mask, unmasked_indices

    def random_tf_masking(self, x, masking_ratio):
        freq_masking_ratio, time_masking_ratio = masking_ratio
        N, L, D = x.shape  # batch, length, dim

        # Calculate the number of patches along frequency and time dimensions
        f_patches = self.image_size[0] // self.patch_size[0]
        t_patches = self.image_size[1] // self.patch_size[1]
        total_patches = f_patches * t_patches

        # Generate 2D grid indices
        indices = torch.arange(total_patches, device=x.device).view(f_patches, t_patches)

        # Generate random row and column masks for the entire batch
        freq_masked_rows = torch.stack([torch.randperm(f_patches, device=x.device)[:int(f_patches * freq_masking_ratio)]
                                        for _ in range(N)])
        time_masked_cols = torch.stack([torch.randperm(t_patches, device=x.device)[:int(t_patches * time_masking_ratio)]
                                        for _ in range(N)])

        # Create the combined masks
        freq_mask = torch.zeros((N, f_patches, t_patches), device=x.device, dtype=bool)
        time_mask = torch.zeros((N, f_patches, t_patches), device=x.device, dtype=bool)

        # Mask rows and columns
        freq_mask.scatter_(1, freq_masked_rows.unsqueeze(-1).expand(-1, -1, t_patches), True)
        time_mask.scatter_(2, time_masked_cols.unsqueeze(1).expand(-1, f_patches, -1), True)

        # Combine the frequency and time masks
        combined_mask = freq_mask | time_mask
        masked_indices = indices.expand(N, -1, -1)[combined_mask].view(N, -1)
        unmasked_indices = indices.expand(N, -1, -1)[~combined_mask].view(N, -1)

        # Create the ids_restore for each sample
        ids_shuffle = torch.cat([unmasked_indices, masked_indices], dim=1)
        ids_restore = torch.zeros_like(ids_shuffle)
        ids_restore.scatter_(1, ids_shuffle, torch.arange(total_patches, device=x.device).expand(N, -1))

        # Gather unmasked tokens
        x_unmasked = torch.gather(x, dim=1, index=unmasked_indices.unsqueeze(-1).expand(-1, -1, D))

        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones((N, total_patches), device=x.device)
        mask.scatter_(1, unmasked_indices, 0)

        return x_unmasked, mask, ids_restore

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if self.masking_method == 'random':
            x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        elif self.masking_method == 'tf_random':
            x, mask, ids_restore = self.random_tf_masking(x, self.mask_ratio)
        else:
            raise ValueError(f"Unknown masking method: {self.masking_method}")

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss, target

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss, target = self.forward_loss(imgs, pred, mask)
        return loss, pred, target, mask


def mae_vit_base_patch16_dec512d1b_debug(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=1, num_heads=1,
        decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=1,
        mlp_ratio=1, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
