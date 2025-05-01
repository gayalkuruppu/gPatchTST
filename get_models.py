from models.patchtst.patchTST import PatchTST
from models.vit.vit import ViT
from models.vit.mae import MAE

import ast

def get_patchTST_model(num_variates, forecast_length, patch_len, stride, num_patch, 
                       n_layers=3, d_model=128, n_heads=16, shared_embedding=True, d_ff=512, norm='BatchNorm', 
                       attn_dropout=0., dropout=0.2, activation='gelu', res_attention=True, pe='zeros', 
                       learn_pe=True, head_dropout=0.2, head_type='pretrain', use_cls_token=False):

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
        use_cls_token=use_cls_token,
    )

    return model


# def get_ViT_MAE_model(image_size=256, patch_size=32, num_classes=1000, dim=1024, depth=6, heads=8, mlp_dim=2048,
#                       masking_ratio=0.75, decoder_dim=512, decoder_depth=6):
#     v = ViT(
#         image_size=image_size,
#         patch_size=patch_size,
#         num_classes=num_classes,
#         dim=dim,
#         depth=depth,
#         heads=heads,
#         mlp_dim=mlp_dim,
#         pool='cls',
#         channels=1,
#     )

#     mae = MAE(
#         encoder=v,
#         masking_ratio=masking_ratio,   # the paper recommended 75% masked patches
#         decoder_dim=decoder_dim,      # paper showed good results with just 512
#         decoder_depth=decoder_depth       # anywhere from 1 to 8
#     )

#     return mae
def get_ViT_MAE_model(image_size=256, patch_size=32, dim=1024, depth=6, heads=8, mlp_dim=2048,
                      decoder_dim=512, decoder_depth=6, head_type='pretrain'):
    v = ViT(
        image_size=image_size,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        pool='cls',
        channels=1,
        head_type='pretrain'
    )

    return v


def get_pretrain_model(model_name, model_config, data_config): # TODO remove data_config
    if model_name == 'patchTST':
        model = get_patchTST_model(
            num_variates=data_config['num_variates'], # TODO get these from model_config
            forecast_length=data_config['forecast_length'],
            patch_len=model_config['patch_len'],
            stride=model_config['stride'],
            num_patch=model_config['num_patch'],
            n_layers=model_config['n_layers'],
            d_model=model_config['d_model'],
            n_heads=model_config['n_heads'],
            shared_embedding=model_config['shared_embedding'],
            d_ff=model_config['d_ff'],
            norm=model_config['norm'],
            attn_dropout=model_config['attn_dropout'],
            dropout=model_config['dropout'],
            activation=model_config['activation'],
            res_attention=model_config['res_attention'],
            pe=model_config['pe'],
            learn_pe=model_config['learn_pe'],
            head_dropout=model_config['head_dropout'],
            head_type=model_config['head_type'],
            use_cls_token=model_config['use_cls_token'],
        )
    elif model_name == 'ViT_MAE':
        model_config['image_size'] = ast.literal_eval(model_config['image_size'])
        model_config['patch_size'] = ast.literal_eval(model_config['patch_size'])
        model = get_ViT_MAE_model(
            image_size=model_config['image_size'], 
            patch_size=model_config['patch_size'], 
            dim=model_config['dim'], 
            depth=model_config['depth'], 
            heads=model_config['heads'], 
            mlp_dim=model_config['mlp_dim'], 
            # masking_ratio=model_config['masking_ratio'], 
            decoder_dim=model_config['decoder_dim'], 
            decoder_depth=model_config['decoder_depth'],
            head_type=model_config['head_type'],
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model
