import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import argparse
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from models.GTM import GTM
from models.FCN import FCN
from utils.data_multitrends import ZeroShotDataset
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from permetrics.regression import Metrics
from pathlib import Path


def cal_error_metrics(gt, forecasts):
    residuals = gt - forecasts

    # Absolute errors
    metr_obj = Metrics(gt, forecasts)
    mae = mean_absolute_error(gt, forecasts)
    wape = 100 * np.sum(np.sum(np.abs(gt - forecasts), axis=-1)) / np.sum(gt)

    #wape = 100 * np.mean(np.sum(np.abs(gt - forecasts), axis=1) / np.sum(gt, axis=1))
    corrs = np.zeros((gt.shape[0],))
    for i in range(gt.shape[0]):
        corrs[i] = np.corrcoef(gt[i, ...], forecasts[i, ...])[0, 1]
    corr = np.nanmean(corrs)
    smape = metr_obj.symmetric_mean_absolute_percentage_error(multi_output=None, decimal=5)


    # Positive and negative errors
    negative_idx = np.where(residuals < 0)
    positive_idx = np.where(residuals >= 0)
    negative_residuals = residuals[negative_idx]
    positive_residuals = residuals[positive_idx]
    mpe = np.sum(positive_residuals)/len(positive_residuals)
    mne = np.sum(negative_residuals)/len(negative_residuals)
    return round(mae, 3), round(smape, 3), round(corr,3), round(mpe, 3), round(mne, 3), round(wape, 3)
    

def print_error_metrics(y_test, y_hat, rescaled_y_test, rescaled_y_hat):
    mae, smape, corr, mpe, mne, wape = cal_error_metrics(y_test, y_hat)
    rescaled_mae, rescaled_smape, rescaled_corr, rescaled_mpe, rescaled_mne, rescaled_wape = cal_error_metrics(rescaled_y_test, rescaled_y_hat)
    print(mae, smape, corr, mpe, mne, wape, rescaled_mae, rescaled_smape, rescaled_corr, rescaled_mpe, rescaled_mne, rescaled_wape)

def run(args):
    print(args)
    
    # Set up CUDA
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')

    # Seeds for reproducibility
    pl.seed_everything(args.seed)

    # Load sales data    
    test_df = pd.read_csv(Path(args.data_folder + 'test.csv'), parse_dates=['release_date'])


     # Load category and color encodings
    cat_dict = torch.load(Path(args.data_folder + 'category_labels.pt'))
    col_dict = torch.load(Path(args.data_folder + 'color_labels.pt'))
    fab_dict = torch.load(Path(args.data_folder + 'fabric_labels.pt'))

    # Load Google trends
    gtrends = pd.read_csv(Path(args.data_folder + 'gtrends.csv'), index_col=[0], parse_dates=True)
    
    test_loader = ZeroShotDataset(test_df, Path(args.data_folder + '/images'), gtrends, cat_dict, col_dict, \
            fab_dict, args.trend_len).get_loader(batch_size=1, train=False)


    model_savename = f'{args.wandb_run}_{args.output_dim}'
    
    # Create model
    model = None
    if args.use_trends:
        args.model_type = 'GTM'
        model = GTM(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_heads=args.num_attn_heads,
            num_layers=args.num_hidden_layers,
            cat_dict=cat_dict,
            col_dict=col_dict,
            tex_dict=tex_dict,
            trend_len=args.trend_len, 
            num_trends= args.num_trends,
            decoder_input_type=args.decoder_input_type,
            use_encoder_mask=args.use_encoder_mask,
            autoregressive=args.autoregressive,
            gpu_num=args.gpu_num
        )
    else:
        args.model_type = 'FCN'
        model = FCN(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            cat_dict=cat_dict,
            col_dict=col_dict,
            tex_dict=tex_dict,
            use_trends=args.use_trends, 
            trend_len=args.trend_len, 
            num_trends= args.num_trends,
            decoder_input_type=args.decoder_input_type,
            use_encoder_mask=args.use_encoder_mask,
            gpu_num=args.gpu_num
        )
    
    model.load_state_dict(torch.load(args.ckpt_path)['state_dict'], strict=False)

    # Forecast the testing set
    model.to(device)
    model.eval()
    gt, forecasts, attns = [], [],[]
    for test_data in tqdm(test_loader, total=len(test_loader), ascii=True):
        with torch.no_grad():
            test_data = [tensor.to(device) for tensor in test_data]
            item_sales, category, color, textures, temporal_features, gtrends, images =  test_data
            y_pred, att = model(category, color,textures, temporal_features, gtrends, images)
            forecasts.append(y_pred.detach().cpu().numpy().flatten()[:args.output_dim])
            gt.append(item_sales.detach().cpu().numpy().flatten()[:args.output_dim])
            attns.append(att.detach().cpu().numpy())

    attns = np.stack(attns)

    forecasts = np.array(forecasts)
    gt = np.array(gt)

    rescale_vals = np.load(args.data_folder + 'normalization_scale.npy')
    rescaled_forecasts = forecasts * rescale_vals
    rescaled_gt = gt * rescale_vals
    print_error_metrics(gt, forecasts, rescaled_gt, rescaled_forecasts)

    
    torch.save({'results': forecasts* rescale_vals, 'gts': gt* rescale_vals, 'codes': item_codes.tolist()}, Path('results/' + model_savename+'.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot sales forecasting')

    # General arguments
    parser.add_argument('--data_folder', type=str, default='dataset/')
    parser.add_argument('--ckpt_path', type=str, default='ckpt/model.pth')
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--seed', type=int, default=21)
    
     # Model specific arguments
    parser.add_argument('--use_trends', type=int, default=1)
    parser.add_argument('--trend_len', type=int, default=52)
    parser.add_argument('--num_trends', type=int, default=3)
    parser.add_argument('--decoder_input_type', type=int, default=3, help='1: Img, 2: Text, 3: Img+Text')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--use_encoder_mask', type=int, default=1)
    parser.add_argument('--autoregressive', type=bool, default=False)
    parser.add_argument('--num_attn_heads', type=int, default=4)
    parser.add_argument('--num_hidden_layers', type=int, default=1)

    parser.add_argument('--wandb_run', type=str, default='Run1')

    args = parser.parse_args()
    run(args)
