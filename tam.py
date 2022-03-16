import interpretdl as it
import paddle
# load vit model and weights
# !wget -c https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams -P assets/
from models.swintransformer import SwinTransformerFineTuning
model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/indoors_geolocation_weights/run/3/model_val_epoch_loss=9.17.ckpt', device_pretrained = 'cpu').eval()

# Call the interpreter.
tam = it.TAMInterpreter(model, use_cuda=False)
img_path = "/hdd2/past_students/virginia/airbnb/images/porto/porto_152.jpg"
heatmap = tam.interpret(
        img_path,
        start_layer=4,
        label=None,  # elephant
        visual=True,
        save_path=None)
heatmap = tam.interpret(
        img_path,
        start_layer=4,
        label=340,  # zebra
        visual=True,
        save_path=None)