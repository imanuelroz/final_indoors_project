# from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
# from mmseg.core.evaluation import get_palette
# config_file = '/home/rozenberg/indoors_geolocation_pycharm/upernet_swin-tiny_patch4_window7_512x512_160k_ade20k.py'
# checkpoint_file = '/home/rozenberg/indoors_geolocation_pycharm/pretrained_models/upernet_swin_tiny_patch4_window7_512x512.pth'
# # build the model from a config file and a checkpoint file
# model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
# print(model)
import torch

from swin_transformer_semantic_segmentation import SwinTransformer

config = dict(
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    ape=False,
    drop_path_rate=0.3,
    patch_norm=True,
    use_checkpoint=False
)
model = SwinTransformer(**config)
print(model)
checkpoint = torch.load(
    '/home/rozenberg/indoors_geolocation_pycharm/pretrained_models/upernet_swin_tiny_patch4_window7_512x512.pth')
print(checkpoint['state_dict'].keys())

checkpoint_backbone = {k.replace('backbone.', ''): v for k, v in checkpoint['state_dict'].items() if
                       k.startswith('backbone')}
model.load_state_dict(checkpoint_backbone)
