import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from vit_pytorch.vit import Transformer

class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_method = 'random', # 'tf_random', 'random
        masking_ratio = 0.75,
        freq_masking_ratio = 0.2,
        time_masking_ratio = 0.2,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_method = masking_method
        self.masking_ratio = masking_ratio
        self.freq_masking_ratio = freq_masking_ratio
        self.time_masking_ratio = time_masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape # [bs x (h_num w_num) x (ph pw)]

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype) 

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        if self.masking_method == 'random':
            num_masked = int(self.masking_ratio * num_patches)
            rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
            masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        elif self.masking_method == 'tf_random':
            freq_patches = self.encoder.image_height // self.encoder.patch_height  # e.g., 10
            time_patches = self.encoder.image_width // self.encoder.patch_width    # e.g., 50

            # Number of full rows and columns to mask
            freq_masked_patches = int(self.freq_masking_ratio * freq_patches)
            time_masked_patches = int(self.time_masking_ratio * time_patches)

            # Randomly select rows and columns to mask
            freq_mask_indices = torch.randperm(freq_patches, device=device)[:freq_masked_patches]
            time_mask_indices = torch.randperm(time_patches, device=device)[:time_masked_patches]

            # Convert to 1D indices (flattened from 2D patch grid)
            masked_rows = freq_mask_indices[:, None] * time_patches  # Shape (freq_masked_patches, 1)
            masked_rows = masked_rows + torch.arange(time_patches, device=device)  # Broadcasted
            masked_rows = masked_rows.flatten()

            masked_cols = torch.arange(freq_patches, device=device)[:, None] * time_patches
            masked_cols = masked_cols + time_mask_indices  # Shape (freq_patches, time_masked_patches)
            masked_cols = masked_cols.flatten()

            # Union of row and column masks
            masked_indices = torch.cat([masked_rows, masked_cols]).unique()
            all_indices = torch.arange(freq_patches * time_patches, device=device)
            unmasked_indices = all_indices[~torch.isin(all_indices, masked_indices)]

            num_masked = masked_indices.shape[0]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens

        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss, pred_pixel_values, masked_patches, patches, masked_indices
