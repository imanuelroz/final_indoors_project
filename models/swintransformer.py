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

class SwinTransformerFineTuning(
       pl.LightningModule):  # LightningModule (è come nn.Module, ma qui devi anche implementare il training step,ex fare backprop su loss etc)

    def __init__(self, chain_classes=94, subregion_classes=23, city_classes=10458,
                 country_classes=183, device_pretrained=None):  # 22,8948,160 struttura del modello
        super().__init__()
        self.lr = 1e-6
        self.p_dropout = 0.1 #poi rimetti a 0.5
        self.save_hyperparameters()
        with open('pretrained_models/swin_transformer/configs/swin_base_patch4_window7_224.yaml') as f:
            model_config = yaml.safe_load(f)
            model_config = model_config['MODEL']['SWIN']
            model_config = {k.lower(): v for k, v in model_config.items()}
        self.pretrained = swin_transformer.SwinTransformer(**model_config, num_classes=21841)
        self.pretrained_path = 'pretrained_models/swin_base_patch4_window7_224_22k.pth'
        self.pretrained.load_state_dict(torch.load(self.pretrained_path, device_pretrained)[
                                            'model'])  #se vuoi usare per ig, metti 'cpu' al posto di device pretrained,  load me lo carica come un dizionario e poi con load_state_dict lo applico al modello pretrained
        self.pretrained.eval()
        self.chain_classes = chain_classes
        self.subregion_classes = subregion_classes
        self.country_classes = country_classes
        self.city_classes = city_classes
        self.dropout = nn.Dropout(self.p_dropout)
        self.finetuner = nn.Sequential(
            nn.Linear(21841, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
        )
        self.subregion_predictor = nn.Linear(500, subregion_classes) #ultimo modello usa 21841
        self.country_predictor = nn.Linear(500, country_classes)  #ultimo modello usa 21841
        #self.city_predictor = nn.Linear(500, city_classes) #21841
        self.city_predictor = nn.Linear(21841, city_classes)
        self.chain_predictor = nn.Linear(500, chain_classes)
        self.test_avg_top_1_accuracy_chain = MeanMetric()
        self.test_avg_top_1_accuracy_city = MeanMetric()
        self.test_avg_top_1_accuracy_country = MeanMetric()
        self.test_avg_top_1_accuracy_subregion = MeanMetric()
        self.test_avg_top_5_accuracy_chain = MeanMetric()
        self.test_avg_top_5_accuracy_city = MeanMetric()
        self.test_avg_top_5_accuracy_country = MeanMetric()
        self.test_avg_top_5_accuracy_subregion = MeanMetric()
        self.test_avg_top_10_accuracy_chain = MeanMetric()
        self.test_avg_top_10_accuracy_city = MeanMetric()
        self.test_avg_top_10_accuracy_country = MeanMetric()
        self.test_avg_top_10_accuracy_subregion = MeanMetric()

    def forward(self, x):
        with torch.no_grad():  # per non fare il backpropag sulla parte freezata
             swin_output = self.pretrained(x)
             swin_output = self.dropout(swin_output)
        output = self.finetuner(swin_output)
        chain_hat = self.chain_predictor(output) #lascia output
        subregion_hat = self.subregion_predictor(output) #l'ultimo modello che stai lanciando ha swin_output
        country_hat = self.country_predictor(output) #l'ultimo modello che stai lanciando ha swin_output
        city_hat = self.city_predictor(swin_output)
        #city_hat = self.city_predictor(output) #se ci metti swin_output invece di output non ci mette gli hidden layers
        return dict(chain_hat=chain_hat, subregion_hat=subregion_hat, country_hat=country_hat,city_hat=city_hat) #,city_hat=city_hat)


    def forward_ig(self, x):
     # per non fare l' integrated gradient
        swin_output = self.pretrained(x)
        swin_output = self.dropout(swin_output)
        output = self.finetuner(swin_output)
        chain_hat = self.chain_predictor(output)
        subregion_hat = self.subregion_predictor(output)
        country_hat = self.country_predictor(output)
        city_hat = self.city_predictor(swin_output)
        #city_hat = self.city_predictor(output) #se ci metti swin_output invece di output non ci mette gli hidden layers
        return dict(chain_hat=chain_hat, subregion_hat=subregion_hat, country_hat=country_hat,city_hat=city_hat) #,city_hat=city_hat)


    def training_step(self, batch, batch_idx):
        x = batch['image']
        #with torch.cuda.amp.autocast():
        #    y_pred = self(x)
        y_pred = self(x)  # stiamo chiamando il metodo call di nn.Module che a sua volta chiama forward
        y_chain = batch['y_chain']
        y_country = batch['y_country']
        y_city = batch['y_city']
        y_subregion = batch['y_subregion']
        loss_values = F.cross_entropy(y_pred['chain_hat'], y_chain)
        loss_values += F.cross_entropy(y_pred['country_hat'], y_country)
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
        y_chain = batch['y_chain']
        y_country = batch['y_country']
        y_city = batch['y_city']
        y_subregion = batch['y_subregion']
        loss_values = F.cross_entropy(y_pred['chain_hat'], y_chain)
        loss_values += F.cross_entropy(y_pred['country_hat'], y_country)
        loss_values += F.cross_entropy(y_pred['city_hat'], y_city)
        loss_values += F.cross_entropy(y_pred['subregion_hat'], y_subregion)
        return dict(loss=loss_values,  y_pred_chain=y_pred['chain_hat'], y_pred_country=y_pred['country_hat'],
                    y_pred_subregion=y_pred['subregion_hat'], y_pred_city=y_pred['city_hat'], **batch)  # aggiungi ,y_pred_city=y_pred['city_hat'] mi spalma anche il batch dentro il dizionario

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
        y_chain = batch['y_chain']
        y_city = batch['y_city']
        y_subregion = batch['y_subregion']
        loss_values = F.cross_entropy(y_pred['chain_hat'], y_chain)
        loss_values += F.cross_entropy(y_pred['country_hat'], y_country)
        loss_values += F.cross_entropy(y_pred['city_hat'], y_city)
        loss_values += F.cross_entropy(y_pred['subregion_hat'], y_subregion)
        return dict(loss=loss_values)

    def on_validation_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.log('val/batch_loss', outputs['loss'].item(), on_step=True) #prog_bar=True

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
        y_pred_chain = output['y_pred_chain'].argsort(descending=True)
        y_pred_country = output['y_pred_country'].argsort(descending=True)
        y_pred_city = output['y_pred_city'].argsort(descending=True)
        y_pred_subregion = output['y_pred_subregion'].argsort(descending=True)
        chain_top_one_accuracy = (y_pred_chain[:, 0] == output['y_chain']).float().mean().item()
        country_top_one_accuracy = (y_pred_country[:, 0] == output['y_country']).float().mean().item()
        city_top_one_accuracy = (y_pred_city[:, 0] == output['y_city']).float().mean().item()
        subregion_top_one_accuracy = (y_pred_subregion[:, 0] == output['y_subregion']).float().mean().item()
        chain_top_five_accuracy = [output['y_chain'][i] in y_pred_chain[i, :5] for i in
                                     range(len(y_pred_chain))]
        chain_top_five_accuracy = sum(chain_top_five_accuracy) / len(chain_top_five_accuracy)
        country_top_five_accuracy = [output['y_country'][i] in y_pred_country[i, :5] for i in
                                     range(len(y_pred_country))]
        country_top_five_accuracy = sum(country_top_five_accuracy) / len(country_top_five_accuracy)

        city_top_five_accuracy = [output['y_city'][i] in y_pred_city[i, :5] for i in range(len(y_pred_city))]
        city_top_five_accuracy = sum(city_top_five_accuracy) / len(city_top_five_accuracy)

        subregion_top_five_accuracy = [output['y_subregion'][i] in y_pred_subregion[i, :5]
                                       for i in range(len(y_pred_subregion))]
        subregion_top_five_accuracy = sum(subregion_top_five_accuracy) / len(subregion_top_five_accuracy)

        chain_top_ten_accuracy = [output['y_chain'][i] in y_pred_chain[i, :10] for i in
                                    range(len(y_pred_chain))]
        chain_top_ten_accuracy = sum(chain_top_ten_accuracy) / len(chain_top_ten_accuracy)

        country_top_ten_accuracy = [output['y_country'][i] in y_pred_country[i, :10] for i in
                                     range(len(y_pred_country))]
        country_top_ten_accuracy = sum(country_top_ten_accuracy) / len(country_top_ten_accuracy)

        city_top_ten_accuracy = [output['y_city'][i] in y_pred_city[i, :10] for i in range(len(y_pred_city))]
        city_top_ten_accuracy = sum(city_top_ten_accuracy) / len(city_top_ten_accuracy)

        subregion_top_ten_accuracy = [output['y_subregion'][i] in y_pred_subregion[i, :10]
                                       for i in range(len(y_pred_subregion))]
        subregion_top_ten_accuracy = sum(subregion_top_ten_accuracy) / len(subregion_top_ten_accuracy)
        self.log('test/loss', output['loss'])
        self.log('test/chain_top_ten_accuracy', chain_top_ten_accuracy)
        self.log('test/country_top_ten_accuracy', country_top_ten_accuracy)
        self.log('test/city_top_ten_accuracy', city_top_ten_accuracy)
        self.log('test/subregion_top_ten_accuracy', subregion_top_ten_accuracy)
        self.log('test/country_top_five_accuracy', country_top_five_accuracy)
        self.log('test/chain_top_five_accuracy', chain_top_five_accuracy)
        self.log('test/city_top_five_accuracy', city_top_five_accuracy)
        self.log('test/subregion_top_five_accuracy', subregion_top_five_accuracy)
        self.log('test/country_top_one_accuracy', country_top_one_accuracy)
        self.log('test/chain_top_one_accuracy', chain_top_one_accuracy)
        self.log('test/city_top_one_accuracy', city_top_one_accuracy)
        self.log('test/subregion_top_one_accuracy', subregion_top_one_accuracy)
        self.test_avg_top_1_accuracy_chain.update(chain_top_one_accuracy)
        self.test_avg_top_1_accuracy_city.update(city_top_one_accuracy)
        self.test_avg_top_1_accuracy_country.update(country_top_one_accuracy)
        self.test_avg_top_1_accuracy_subregion.update(subregion_top_one_accuracy)
        self.test_avg_top_5_accuracy_chain.update(chain_top_five_accuracy)
        self.test_avg_top_5_accuracy_city.update(city_top_five_accuracy)
        self.test_avg_top_5_accuracy_country.update(country_top_five_accuracy)
        self.test_avg_top_5_accuracy_subregion.update(subregion_top_five_accuracy)
        self.test_avg_top_10_accuracy_chain.update(chain_top_ten_accuracy)
        self.test_avg_top_10_accuracy_city.update(city_top_ten_accuracy)
        self.test_avg_top_10_accuracy_country.update(country_top_ten_accuracy)
        self.test_avg_top_10_accuracy_subregion.update(subregion_top_ten_accuracy)

