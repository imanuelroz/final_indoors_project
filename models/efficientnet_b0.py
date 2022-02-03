import torchmetrics
import torchvision.models as models
import torch
import yaml
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from typing import Any, Optional
from torch.nn import functional as F
from pretrained_models.swin_transformer.models import build
from pretrained_models.swin_transformer.models import swin_transformer
import pytorch_lightning as pl
from metriche import MeanMetric
from efficientnet_pytorch import EfficientNet
#mobilenet = models.mobilenet_v2(pretrained=True)
#mnasnet = models.mnasnet1_0(pretrained=True)


#model = EfficientNet.from_pretrained('efficientnet-b0')



class EfficientNet_FineTuning(pl.LightningModule):
    # transfer learning if pretrained=True
    def __init__(self, subregion_classes=23, city_classes=10458, country_classes=183):
        super().__init__()
        # log hyperparameters
        self.save_hyperparameters()
        self.lr = 1e-6
        self.p_dropout = 0.5
        #self.dim = input_shape
        self.subregion_classes = subregion_classes
        self.city_classes = city_classes
        self.country_classes = country_classes
        self.feature_extractor = models.efficientnet_b0(pretrained=True).features
        # layers are frozen by using eval()
        self.feature_extractor.eval()
        # freeze params
        self.subregion_classes = subregion_classes
        self.country_classes = country_classes
        #self.city_classes = city_classes
        self.dropout = nn.Dropout(self.p_dropout)
        self.finetuner = nn.Sequential(
            nn.Linear(62720, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
        )
        #self.model.fc = torch.nn.Linear(model.fc.in_features, subregion_classes, city_classes, country_classes)
        self.subregion_predictor = nn.Linear(500, subregion_classes)
        self.country_predictor = nn.Linear(500, country_classes)
        self.city_predictor = nn.Linear(500, city_classes)
        self.test_avg_top_1_accuracy_city = MeanMetric()
        self.test_avg_top_1_accuracy_country = MeanMetric()
        self.test_avg_top_1_accuracy_subregion = MeanMetric()
        self.test_avg_top_5_accuracy_city = MeanMetric()
        self.test_avg_top_5_accuracy_country = MeanMetric()
        self.test_avg_top_5_accuracy_subregion = MeanMetric()
        self.test_avg_top_10_accuracy_city = MeanMetric()
        self.test_avg_top_10_accuracy_country = MeanMetric()
        self.test_avg_top_10_accuracy_subregion = MeanMetric()

    def forward(self, x):
        with torch.no_grad():  # per non fare il backpropag sulla parte freezata
            model_output = self.feature_extractor(x).flatten(start_dim=1)
            model_output = self.dropout(model_output)
        output = self.finetuner(model_output)
        subregion_hat = self.subregion_predictor(output)
        country_hat = self.country_predictor(output)
        city_hat = self.city_predictor(output)
        return dict(subregion_hat=subregion_hat, country_hat=country_hat, city_hat=city_hat)

    def training_step(self, batch, batch_idx):
        x = batch['image']
        #with torch.cuda.amp.autocast():
        #    y_pred = self(x)
        y_pred = self(x)  # stiamo chiamando il metodo call di nn.Module che a sua volta chiama forward
        y_country = batch['y_country']
        y_city = batch['y_city']
        y_subregion = batch['y_subregion']
        loss_values = F.cross_entropy(y_pred['country_hat'], y_country)
        loss_values += F.cross_entropy(y_pred['city_hat'], y_city)
        loss_values += F.cross_entropy(y_pred['subregion_hat'], y_subregion)
        return dict(loss=loss_values)
        #return dict(loss=loss_values, y_pred_country=y_pred['country_hat'], y_pred_city=y_pred['city_hat'],
                    #y_pred_subregion=y_pred['subregion_hat'], **batch)  # mi spalma anche il batch dentro il dizionario

    def _full_pred_step(self, batch, batch_idx):
        x = batch['image']
        #with torch.cuda.amp.autocast():
        #    y_pred = self(x)
        y_pred = self(x)  # stiamo chiamando il metodo call di nn.Module che a sua volta chiama forward
        y_country = batch['y_country']
        y_city = batch['y_city']
        y_subregion = batch['y_subregion']
        loss_values = F.cross_entropy(y_pred['country_hat'], y_country)
        loss_values += F.cross_entropy(y_pred['city_hat'], y_city)
        loss_values += F.cross_entropy(y_pred['subregion_hat'], y_subregion)
        return dict(loss=loss_values, y_pred_country=y_pred['country_hat'],
                    y_pred_subregion=y_pred['subregion_hat'], y_pred_city=y_pred['city_hat'], **batch)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.log('train/batch_loss', outputs['loss'].item(), on_step=True) #prog_bar=True

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        #with torch.cuda.amp.autocast():
        #    y_pred = self(x)
        y_pred = self(x)  # stiamo chiamando il metodo call di nn.Module che a sua volta chiama forward
        y_country = batch['y_country']
        y_city = batch['y_city']
        y_subregion = batch['y_subregion']
        loss_values = F.cross_entropy(y_pred['country_hat'], y_country)
        loss_values += F.cross_entropy(y_pred['city_hat'], y_city)
        loss_values += F.cross_entropy(y_pred['subregion_hat'], y_subregion)
        return dict(loss=loss_values)

    def on_validation_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.log('val/batch_loss', outputs['loss'].item(), on_step=True)

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        mean_loss = torch.stack(
            [o['loss'] for o in outputs]).mean().item()  # con torch.stack lo trasformo in un tensore pytorch
        self.log('train/epoch_loss', mean_loss, on_epoch=True, on_step=False)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        mean_loss = torch.stack(
            [o['loss'] for o in outputs]).mean().item()  # con torch.stack lo trasformo in un tensore pytorch
        self.log('val_epoch_loss', mean_loss) #ho tolto on epoch = True

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        output = self._full_pred_step(batch, batch_idx)
        y_pred_country = output['y_pred_country'].argsort(descending=True)
        y_pred_city = output['y_pred_city'].argsort(descending=True)
        y_pred_subregion = output['y_pred_subregion'].argsort(descending=True)
        country_top_one_accuracy = (y_pred_country[:, 0] == output['y_country']).float().mean().item()
        city_top_one_accuracy = (y_pred_city[:, 0] == output['y_city']).float().mean().item()
        subregion_top_one_accuracy = (y_pred_subregion[:, 0] == output['y_subregion']).float().mean().item()
        country_top_five_accuracy = [output['y_country'][i] in y_pred_country[i, :5] for i in
                                     range(len(y_pred_country))]
        country_top_five_accuracy = sum(country_top_five_accuracy) / len(country_top_five_accuracy)

        city_top_five_accuracy = [output['y_city'][i] in y_pred_city[i, :5] for i in range(len(y_pred_city))]
        city_top_five_accuracy = sum(city_top_five_accuracy) / len(city_top_five_accuracy)

        subregion_top_five_accuracy = [output['y_subregion'][i] in y_pred_subregion[i, :5]
                                       for i in range(len(y_pred_subregion))]
        subregion_top_five_accuracy = sum(subregion_top_five_accuracy) / len(subregion_top_five_accuracy)


        country_top_ten_accuracy = [output['y_country'][i] in y_pred_country[i, :10] for i in
                                     range(len(y_pred_country))]
        country_top_ten_accuracy = sum(country_top_ten_accuracy) / len(country_top_ten_accuracy)

        city_top_ten_accuracy = [output['y_city'][i] in y_pred_city[i, :10] for i in range(len(y_pred_city))]
        city_top_ten_accuracy = sum(city_top_ten_accuracy) / len(city_top_ten_accuracy)

        subregion_top_ten_accuracy = [output['y_subregion'][i] in y_pred_subregion[i, :10]
                                       for i in range(len(y_pred_subregion))]
        subregion_top_ten_accuracy = sum(subregion_top_ten_accuracy) / len(subregion_top_ten_accuracy)
        self.log('test/loss', output['loss'])
        self.log('test/country_top_ten_accuracy', country_top_ten_accuracy)
        self.log('test/city_top_ten_accuracy', city_top_ten_accuracy)
        self.log('test/subregion_top_ten_accuracy', subregion_top_ten_accuracy)
        self.log('test/country_top_five_accuracy', country_top_five_accuracy)
        self.log('test/city_top_five_accuracy', city_top_five_accuracy)
        self.log('test/subregion_top_five_accuracy', subregion_top_five_accuracy)
        self.log('test/country_top_one_accuracy', country_top_one_accuracy)
        self.log('test/city_top_one_accuracy', city_top_one_accuracy)
        self.log('test/subregion_top_one_accuracy', subregion_top_one_accuracy)
        self.test_avg_top_1_accuracy_city.update(city_top_one_accuracy)
        self.test_avg_top_1_accuracy_country.update(country_top_one_accuracy)
        self.test_avg_top_1_accuracy_subregion.update(subregion_top_one_accuracy)
        self.test_avg_top_5_accuracy_city.update(city_top_five_accuracy)
        self.test_avg_top_5_accuracy_country.update(country_top_five_accuracy)
        self.test_avg_top_5_accuracy_subregion.update(subregion_top_five_accuracy)
        self.test_avg_top_10_accuracy_city.update(city_top_ten_accuracy)
        self.test_avg_top_10_accuracy_country.update(country_top_ten_accuracy)
        self.test_avg_top_10_accuracy_subregion.update(subregion_top_ten_accuracy)

