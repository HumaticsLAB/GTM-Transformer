import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import pipeline
from torchvision import models
from fairseq.optim.adafactor import Adafactor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=52):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TimeDistributed(nn.Module):
    # Takes any module and stacks the time dimension with the batch dimenison of inputs before applying the module
    # Insipired from https://keras.io/api/layers/recurrent_layers/time_distributed/
    # https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module # Can be any layer we wish to apply like Linear, Conv etc
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class FusionNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, use_img, use_text, dropout=0.2):
        super(FusionNetwork, self).__init__()
        
        self.img_pool = nn.AdaptiveAvgPool2d((1,1))
        self.img_linear = nn.Linear(2048, embedding_dim)
        self.use_img = use_img
        self.use_text = use_text
        input_dim = embedding_dim + (embedding_dim*use_img) + (embedding_dim*use_text)
        self.feature_fusion = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim)
        )

    def forward(self, img_encoding, text_encoding, dummy_encoding):
        # Fuse static features together
        pooled_img = self.img_pool(img_encoding)
        condensed_img = self.img_linear(pooled_img.flatten(1))

        # Build input
        decoder_inputs = []
        if self.use_img == 1:
            decoder_inputs.append(condensed_img) 
        if self.use_text == 1:
            decoder_inputs.append(text_encoding) 
        decoder_inputs.append(dummy_encoding)
        concat_features = torch.cat(decoder_inputs, dim=1)

        final = self.feature_fusion(concat_features)

        return final

class GTrendEmbedder(nn.Module):
    def __init__(self, forecast_horizon, embedding_dim, use_mask, trend_len, num_trends,  gpu_num):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.input_linear = TimeDistributed(nn.Linear(num_trends, embedding_dim))
        self.pos_embedding = PositionalEncoding(embedding_dim, max_len=trend_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.use_mask = use_mask
        self.gpu_num = gpu_num

    def _generate_encoder_mask(self, size, forecast_horizon):
        mask = torch.zeros((size, size))
        split = math.gcd(size, forecast_horizon)
        for i in range(0, size, split):
            mask[i:i+split, i:i+split] = 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:'+str(self.gpu_num))
        return mask
    
    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:'+str(self.gpu_num))
        return mask

    def forward(self, gtrends):
        gtrend_emb = self.input_linear(gtrends.permute(0,2,1))
        gtrend_emb = self.pos_embedding(gtrend_emb.permute(1,0,2))
        input_mask = self._generate_encoder_mask(gtrend_emb.shape[0], self.forecast_horizon)
        if self.use_mask == 1:
            gtrend_emb = self.encoder(gtrend_emb, input_mask)
        else:
            gtrend_emb = self.encoder(gtrend_emb)
        return gtrend_emb
        
class TextEmbedder(nn.Module):
    def __init__(self, embedding_dim, cat_dict, col_dict, fab_dict, gpu_num):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cat_dict = {v: k for k, v in cat_dict.items()}
        self.col_dict = {v: k for k, v in col_dict.items()}
        self.fab_dict = {v: k for k, v in fab_dict.items()}
        self.word_embedder = pipeline('feature-extraction', model='bert-base-uncased')
        self.fc = nn.Linear(768, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.gpu_num = gpu_num

    def forward(self, category, color, fabric):
        textual_description = [self.col_dict[color.detach().cpu().numpy().tolist()[i]] + ' ' \
                + self.fab_dict[fabric.detach().cpu().numpy().tolist()[i]] + ' ' \
                + self.cat_dict[category.detach().cpu().numpy().tolist()[i]] for i in range(len(category))]


        # Use BERT to extract features
        word_embeddings = self.word_embedder(textual_description)

        # BERT gives us embeddings for [CLS] ..  [EOS], which is why we only average the embeddings in the range [1:-1] 
        # We're not fine tuning BERT and we don't want the noise coming from [CLS] or [EOS]
        word_embeddings = [torch.FloatTensor(x[1:-1]).mean(axis=0) for x in word_embeddings] 
        word_embeddings = torch.stack(word_embeddings).to('cuda:'+str(self.gpu_num))
        
        # Embed to our embedding space
        word_embeddings = self.dropout(self.fc(word_embeddings))

        return word_embeddings

class ImageEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        # Img feature extraction
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False

        # Fine tune resnet
        # for c in list(self.resnet.children())[6:]:
        #     for p in c.parameters():
        #         p.requires_grad = True
        
    def forward(self, images):        
        img_embeddings = self.resnet(images)  
        size = img_embeddings.size()
        out = img_embeddings.view(*size[:2],-1)

        return out.view(*size).contiguous() # batch_size, 2048, image_size/32, image_size/32

class DummyEmbedder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.day_embedding = nn.Linear(1, embedding_dim)
        self.week_embedding = nn.Linear(1, embedding_dim)
        self.month_embedding = nn.Linear(1, embedding_dim)
        self.year_embedding = nn.Linear(1, embedding_dim)
        self.dummy_fusion = nn.Linear(embedding_dim*4, embedding_dim)
        self.dropout = nn.Dropout(0.2)


    def forward(self, temporal_features):
        # Temporal dummy variables (day, week, month, year)
        d, w, m, y = temporal_features[:, 0].unsqueeze(1), temporal_features[:, 1].unsqueeze(1), \
            temporal_features[:, 2].unsqueeze(1), temporal_features[:, 3].unsqueeze(1)
        d_emb, w_emb, m_emb, y_emb = self.day_embedding(d), self.week_embedding(w), self.month_embedding(m), self.year_embedding(y)
        temporal_embeddings = self.dummy_fusion(torch.cat([d_emb, w_emb, m_emb, y_emb], dim=1))
        temporal_embeddings = self.dropout(temporal_embeddings)

        return temporal_embeddings

class FCN(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, output_dim, cat_dict, col_dict, tex_dict, \
        use_trends, use_text, use_img, trend_len, num_trends, use_encoder_mask=1, gpu_num=2):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_len = output_dim
        self.use_encoder_mask = use_encoder_mask
        self.gpu_num = gpu_num
        self.use_trends = use_trends
        self.save_hyperparameters()

         # Encoder
        self.dummy_encoder = DummyEmbedder(embedding_dim)
        self.image_encoder = ImageEmbedder()
        self.text_encoder = TextEmbedder(embedding_dim, cat_dict, col_dict, tex_dict, gpu_num)
        self.gtrend_encoder = GTrendEmbedder(output_dim, hidden_dim, use_encoder_mask, trend_len, num_trends, gpu_num)
        self.static_feature_encoder = FusionNetwork(embedding_dim, hidden_dim, use_img, use_text)

        # Decoder
        decoder_in = hidden_dim + (use_trends*(trend_len*hidden_dim))
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim*4, self.output_len)
        )

    def forward(self, category, color, fabric, temporal_features, gtrends, images):
        # Encode features and get inputs
        img_encoding = self.image_encoder(images)
        dummy_encoding = self.dummy_encoder(temporal_features)
        text_encoding = self.text_encoder(category, color, fabric)
        gtrend_encoding = self.gtrend_encoder(gtrends)

        # Fuse static features together
        static_feature_fusion = self.static_feature_encoder(img_encoding, text_encoding, dummy_encoding)

        # Decode
        if self.use_trends == 1:
            tgt = torch.cat([static_feature_fusion, gtrend_encoding.reshape(static_feature_fusion.shape[0], -1)], dim=-1)
        else:
            tgt = static_feature_fusion

        forecast = self.decoder(tgt)

        return forecast.view(-1, self.output_len)

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        item_sales, category, color, fabric, temporal_features, gtrends, images = train_batch 
        forecasted_sales = self.forward(category, color, fabric, temporal_features, gtrends, images)
        forecasting_loss = F.mse_loss(item_sales, forecasted_sales.squeeze())
        loss = forecasting_loss
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        item_sales, category, color, fabric, temporal_features, gtrends, images = val_batch 
        forecasted_sales = self.forward(category, color, fabric, temporal_features, gtrends, images)
        
        return item_sales.squeeze(), forecasted_sales.squeeze()

    def validation_epoch_end(self, val_step_outputs):
        item_sales, forecasted_sales = [x[0] for x in val_step_outputs], [x[1] for x in val_step_outputs]
        item_sales, forecasted_sales = torch.stack(item_sales), torch.stack(forecasted_sales)
        rescaled_item_sales, rescaled_forecasted_sales = item_sales*1065, forecasted_sales*1065 # 1065 is the normalization factor (max of the sales of the training set)
        mae = F.l1_loss(rescaled_item_sales, rescaled_forecasted_sales)
        loss = F.mse_loss(item_sales, forecasted_sales.squeeze())
        self.log('val_loss', loss)
        self.log('val_mae', mae)
        print('Validation MAE:', mae.detach().cpu().numpy(), 'LR:', self.optimizers().param_groups[0]['lr'])