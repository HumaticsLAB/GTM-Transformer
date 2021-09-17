import os
import argparse
from pickle import TRUE
from pytorch_lightning.loggers import wandb
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
from datetime import datetime
from models.GTMDisjoint import GTM
from models.FCN import FCN
from utils.data_multitrends import ZeroShotDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run(args):
    print(args)
    # Seeds for reproducibility
    pl.seed_everything(args.seed)

    # Load sales data
    train_df = pd.read_csv(Path(args.data_folder + 'train.csv'), parse_dates=['release_date'])
    test_df = pd.read_csv(Path(args.data_folder + 'test.csv'), parse_dates=['release_date'])
    # train_df = train_df.iloc[:128, :]
    # test_df = test_df.iloc[:128, :]

     # Load category and color encodings
    cat_dict = torch.load(Path(args.data_folder + 'category_labels.pt'))
    col_dict = torch.load(Path(args.data_folder + 'color_labels.pt'))
    fab_dict = torch.load(Path(args.data_folder + 'fabric_labels.pt'))

    # Load Google trends
    gtrends = pd.read_csv(Path(args.data_folder + 'gtrends.csv'), index_col=[0], parse_dates=True)
    
    train_loader = ZeroShotDataset(train_df, Path(args.data_folder + '/images'), gtrends, cat_dict, col_dict,  \
            fab_dict, args.trend_len).get_loader(batch_size=args.batch_size, train=True)
    test_loader = ZeroShotDataset(test_df, Path(args.data_folder + '/images'), gtrends, cat_dict, col_dict, \
            fab_dict, args.trend_len).get_loader(batch_size=1, train=False)

    
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


    # Model Training
    # Define model saving procedure
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    model_savename = args.model_type + '_' + args.wandb_run

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.log_dir + '/'+args.model_type,
        filename=model_savename+'---{epoch}---'+dt_string,
        monitor='val_mae',
        mode='min',
        save_top_k=1
    )
    import wandb
    wandb.init(entity=args.wandb_entity, project=args.wandb_proj, name=args.wandb_run)
    wandb_logger = pl_loggers.WandbLogger()
    
    wandb_logger.watch(model)
    

    # tb_logger = pl_loggers.TensorBoardLogger('lightning_logs/', name=model_savename)
    trainer = pl.Trainer(gpus=[args.gpu_num], max_epochs=args.epochs, check_val_every_n_epoch=5, \
                        logger=wandb_logger, callbacks=[checkpoint_callback])
    
    # Fit model
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=test_loader)

    # Save final model to another directory where the fully trained models are kept
    trainer.save_checkpoint(args.ckpt_dir + model_savename + '.ckpt')

    # Print out path of best model
    print(checkpoint_callback.best_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot sales forecasting')

     # General arguments
    parser.add_argument('--data_folder', type=str, default='dataset/')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--gpu_num', type=int, default=0)

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

    # Model wandb arguments
    parser.add_argument('--wandb_entity', type=str, default='...')
    parser.add_argument('--wandb_proj', type=str, default='GTM')
    parser.add_argument('--wandb_run', type=str, default='Run1')

    args = parser.parse_args()
    run(args)
