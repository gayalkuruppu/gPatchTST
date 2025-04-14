from models.patchtst.patchTST import PatchTST

def get_patchTST_model(num_variates, forecast_length, patch_len, stride, num_patch, 
                       n_layers=3, d_model=128, n_heads=16, shared_embedding=True, d_ff=512, norm='BatchNorm', 
                       attn_dropout=0., dropout=0.2, activation='gelu', res_attention=True, pe='zeros', 
                       learn_pe=True, head_dropout=0.2, head_type='pretrain'):

    model = PatchTST(
        c_in=num_variates,
        target_dim=forecast_length,
        patch_len=patch_len,
        stride=stride,
        num_patch=num_patch,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        shared_embedding=shared_embedding, # for each variate
        d_ff=d_ff,
        norm=norm,
        attn_dropout=attn_dropout, # in scaled dot product attention only
        dropout=dropout, # dropout for encoder and each layer
        act=activation,
        res_attention=res_attention,
        pe=pe,
        learn_pe=learn_pe,
        head_dropout=head_dropout,
        head_type=head_type, # ['pretrain', 'prediction', 'regression', 'classification']
    )

    return model