import argparse
import random
from data_provider.data_factory import data_provider
from utils.metrics import metric
import torch
import os
from models import TimeMixerPP
import warnings
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('./models')

fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='TimeMixerPP')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=False, default=0, help='status')
parser.add_argument('--model_id', type=str, required=True, default='solar_96_96', help='model id')
parser.add_argument('--model', type=str, required=True, default='TimeMixerPP',
                    help='model name, options: [TimeMixerPP]')

# data loader
parser.add_argument('--data', type=str, required=True, default='Solar', help='dataset type')
parser.add_argument('--root_path', type=str, required=True, default='./dataset/solar/', help='root path of the data file')
parser.add_argument('--data_path', type=str, required=True, default='solar_AL.txt', help='data file')
parser.add_argument('--test_data_path', type=str, required=False, default='weather.csv',
                    help='test data file used in zero shot forecasting')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--test_seq_len', type=int, default=672, help='test seq len')
parser.add_argument('--test_label_len', type=int, default=576, help='test label len')
parser.add_argument('--test_pred_len', type=int, default=96, help='test pred len')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

# model define
parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=137, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=137, help='decoder input size')
parser.add_argument('--c_out', type=int, default=137, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='dft_decomp',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=2, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default='avg',
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--channel_mixing', type=int, default=1,
                    help='0: channel mixing 1: whether to use channel_mixing')

parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
parser.add_argument('--percent', type=int, default=100, help='few shot percent')

# # optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
parser.add_argument('--des', type=str, default='Exp', help='exp description')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--comment', type=str, default='none', help='com')

parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
# parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')



if __name__ == '__main__':

    args = parser.parse_args()
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 4:
            devices = '0,1,2,3'
        elif num_gpus == 8:
            devices = '0,1,2,3,4,5,6,7'
        elif num_gpus == 12:
            devices = '0,1,2,3,4,5,6,7,8,9,10,11'
        elif num_gpus == 16:
            devices = '0,1,2,3,4,5,6,7,8,9,10,11,12,14,14,15'
        elif num_gpus == 1:
            devices = '0'
        args.devices = devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)


    if args.is_training:
        print("training")
    else:
        ii = 0
        setting = '{}_{}_{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.comment,
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        if args.task_name == 'long_term_forecast':
            if args.use_gpu:
                import platform

                if platform.system() == 'Darwin':
                    device = torch.device('mps')
                    print('Use MPS')
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    args.gpu) if not args.use_multi_gpu else args.devices
                device = torch.device('cuda:{}'.format(args.gpu))
                if args.use_multi_gpu:
                    print('Use GPU: cuda{}'.format(args.device_ids))
                else:
                    print('Use GPU: cuda:{}'.format(args.gpu))
            else:
                device = torch.device('cpu')
                print('Use CPU')

            test_data, test_loader = data_provider(args, flag='test')

            try:
                model = torch.load('./ckpt/all_checkpoint.pth',
                                            map_location=device, weights_only=False)[args.model_id]
            except:
                print("Your model_id is not found.")

            model = model.float()
            preds = []
            trues = []
            folder_path = './test_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                        tqdm(test_loader, desc="Testing Progress")):
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)

                    batch_x_mark = batch_x_mark.float().to(device)
                    batch_y_mark = batch_y_mark.float().to(device)
                          
                    if 'PEMS' == args.data or 'Solar' == args.data:
                        batch_x_mark = None
                        batch_y_mark = None

                    if args.down_sampling_layers == 0:
                        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                            device)
                    else:
                        dec_inp = None

                    # encoder - decoder
                    if args.use_amp:
                        with torch.cuda.amp.autocast():
                            if args.output_attention:
                                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if args.output_attention:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if args.features == 'MS' else 0

                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                    pred = outputs
                    true = batch_y

                    preds.append(pred)
                    trues.append(true)

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mse:{}, mae:{}'.format(mse, mae))
            print('rmse:{}, mape:{}, mspe:{}'.format(rmse, mape, mspe))

            f = open("result_long_term_forecast.txt", 'a')
            f.write(setting + "  \n")
            if args.data == 'PEMS':
                f.write('mae:{}, mape:{}, rmse:{}'.format(mae, mape, rmse))
            else:
                f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n')
            f.write('\n')
            f.close()

            torch.cuda.empty_cache()
