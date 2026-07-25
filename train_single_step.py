import argparse
import math
import time

import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import importlib

from util import *
from trainer import Optim

my_data ='./data/electricity.txt'
my_data_name = 'electricity'
my_save_path = './save/electricity' # save path of model's parameters
my_result_path = './save/result/electricity' # save path of results
my_num_node = 321 # have to be changed according to dataset  exchange_rate:8, solar:137, traffic:862, electricity:321
my_topk = 20 # have to be changed according to dataset  exchange_rate:4, solar:20, traffic:20, electricity:20
my_train = True
my_new_graph_method = False
my_runs = 10
my_epochs = 30


my_device = 'cuda'

my_seq_in_len = 24*7
my_seq_out_len = 1 # single step prediction에서는 1로 고정?
# my_horizon = 3   밑에 코드 실행단에서 모든 horizon에 대해 실행하도록 코드 바꿈  # single step prediction에서 horizon 3,6,9,12,24 이런건 해당 변수를 의미
my_batch_size = 32

use_l1loss = True

my_dilation_exponential = 2
my_in_dim = 1 # use only velocity
my_hidden_dim = 32
my_layer_depth = 3
my_using_only_TC = False # True: only use TC, False: use TC + GCN when making new graph learning method.

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        if len(output.shape)==1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.m)

    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return rse, rae, correlation


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]
            id = torch.tensor(id).to(device)
            tx = X[:, :, id.long(), :]
            ty = Y[:, id.long()]
            output = model(tx,id)
            output = torch.squeeze(output)
            scale = data.scale.expand(output.size(0), data.m)
            scale = scale[:,id.long()]
            loss = criterion(output * scale, ty * scale)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * data.m)
            grad_norm = optim.step()

        if iter%100==0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter,loss.item()/(output.size(0) * data.m)))
        iter += 1
    return total_loss / n_samples



parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--train', type=str_to_bool, default=my_train ,help='whether to do training or testing')
parser.add_argument('--data', type=str, default=my_data,
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default=my_save_path,
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=use_l1loss)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device',type=str,default=my_device,help='')
parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=my_num_node,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=my_topk,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=my_dilation_exponential,help='dilation exponential')
parser.add_argument('--conv_channels',type=int,default=16,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=16,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=32,help='skip channels')
parser.add_argument('--end_channels',type=int,default=64,help='end channels')
parser.add_argument('--in_dim',type=int,default=my_in_dim,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=my_seq_in_len,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=my_seq_out_len,help='output sequence length')
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--layers',type=int,default=5,help='number of layers')
parser.add_argument('--hidden_dim',type=int,default=my_hidden_dim,help='hidden state dimension of new graph learning layer')
parser.add_argument('--layer_depth',type=int,default=my_layer_depth,help='depth of new graph learning layer')
parser.add_argument('--new_graph_learning', type=str_to_bool, default=my_new_graph_method ,help='whether to do new graph learning method or not')
parser.add_argument('--new_graph_only_TC', type=str_to_bool, default=my_using_only_TC ,help='True: only use TC, False: use TC + GCN when making new graph learning method.')

parser.add_argument('--batch_size',type=int,default=my_batch_size,help='batch size')
parser.add_argument('--lr',type=float,default=0.0001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')

parser.add_argument('--clip',type=int,default=5,help='clip')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')

parser.add_argument('--epochs',type=int,default=my_epochs,help='')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs') # sub graph 개수
parser.add_argument('--step_size',type=int,default=100,help='step_size')
parser.add_argument('--runs',type=int,default= my_runs ,help='number of runs')


args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)

def main(runid):

    Data = DataLoaderS(args.data, 0.6, 0.2, device, args.horizon, args.seq_in_len, args.normalize)

    model = gtnet(gcn_true = args.gcn_true,
                  buildA_true = args.buildA_true,
                  gcn_depth = args.gcn_depth,
                  num_nodes = args.num_nodes,
                  device = device,
                  predefined_A = None, # for single step
                  hidden_channels=args.hidden_dim,
                  seq_length=args.seq_in_len,
                  layer_depth=args.layer_depth,
                  dropout=args.dropout,subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim,
                  new_graph_learning=args.new_graph_learning,
                  new_graph_only_TC=args.new_graph_only_TC,
                  dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False)

    model = model.to(device)

    print(args)
    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False).to(device)
    else:
        criterion = nn.MSELoss(size_average=False).to(device)
    evaluateL2 = nn.MSELoss(size_average=False).to(device)
    evaluateL1 = nn.L1Loss(size_average=False).to(device)


    best_val = 10000000
    optim = Optim(
        model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
    )

    if args.train:
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            print('begin training')

            his_train_loss = []
            his_val_acc = []
            his_val_rae = []
            his_val_corr = []

            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()
                train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
                val_loss, val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                                   args.batch_size)
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                        epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr), flush=True)
                # Save the model if the validation loss is the best we've seen so far.

                his_train_loss.append(train_loss)
                his_val_acc.append(val_loss)
                his_val_rae.append(val_rae)
                his_val_corr.append(val_corr)

                if val_loss < best_val:
                    os.makedirs(args.save, exist_ok=True)
                    # when previous version is already existed, we will save the new version as "expid-runid-2.pth"
                    torch.save(model.state_dict(), args.save + f"/{my_data_name}_horizon{args.horizon}" + "_" + str(runid) + ".pth")
                    best_val = val_loss

            col_name = ['train_loss', 'val_acc', 'val_rae', 'val_corr']

            loss_data = np.concatenate(([his_train_loss],[his_val_acc],[his_val_rae],[his_val_corr]),axis=0).T

            ## store every epoch loss
            datetime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            write_csv(my_result_path,f'every_epoch_train_valid_loss_horizon{args.horizon}_run{runid+1}_{datetime}.csv',loss_data,col_name)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    # test
    # Load the best saved model.
    model.load_state_dict(torch.load(args.save + f"/{my_data_name}_horizon{args.horizon}" + "_" + str(runid) + ".pth"))

    vtest_acc, vtest_rae, vtest_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                         args.batch_size)
    test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                         args.batch_size)
    print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
    return vtest_acc, vtest_rae, vtest_corr, test_acc, test_rae, test_corr

if __name__ == "__main__":

    for horizon in [3]: # [3,6,12,24]:
        args.horizon = horizon

        vacc = []
        vrae = []
        vcorr = []
        acc = []
        rae = []
        corr = []
        for i in range(args.runs):
            val_acc, val_rae, val_corr, test_acc, test_rae, test_corr = main(i)
            vacc.append(val_acc)
            vrae.append(val_rae)
            vcorr.append(val_corr)
            acc.append(test_acc)
            rae.append(test_rae)
            corr.append(test_corr)

        col_name = ['test_acc', 'test_rae', 'test_corr']

        loss_data = np.concatenate(([acc], [rae], [corr]), axis=0).T

        ## store every epoch loss
        datetime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        write_csv(my_result_path, f'every_test_result_for_horizon{args.horizon}_run{my_runs}_{datetime}.csv',
                  loss_data, col_name)

        print('\n\n')
        print(f'{args.horizon}horizon_{args.runs}runs average')
        print('\n\n')
        print("valid\trse\trae\tcorr")
        print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae), np.mean(vcorr)))
        print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae), np.std(vcorr)))
        print('\n\n')
        print("test\trse\trae\tcorr")
        print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae), np.mean(corr)))
        print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae), np.std(corr)))



