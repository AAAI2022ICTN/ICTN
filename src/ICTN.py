import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
import data_got
import numpy as np
import torch.utils.data as data_utils
from utils import Bar, Logger, AverageMeter, precision_k, calc_acc,ndcg_k, calc_f1, mkdir_p, savefig
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=120, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('-c', '--checkpoint', default='imprint_checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: imprint_checkpoint)')
parser.add_argument('--model', default='base_checkpoint/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to model (default: none)')
parser.add_argument('--random', action='store_true', help='whether use random novel weights')
parser.add_argument('--num_sample', default=8, type=int,
                    metavar='N', help='number of novel sample (default: 1)')
parser.add_argument('--tnum_sample', default=5, type=int,
                    metavar='N', help='number of novel sample (default: 1)')
parser.add_argument('--test-novel-only', action='store_true', help='whether only test on novel classes')
parser.add_argument('--aug', action='store_true', help='whether use data augmentation during training')
parser.add_argument('--lstm_hid_dim', default=150, type=int, metavar='N',
                    help='lstm_hid_dim')
parser.add_argument('--num_class', default=30, type=int, metavar='N',
                    help='the number of class')
parser.add_argument('--epochs', default=10,type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--samp_freq', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--d_a', default=100, type=int, metavar='N',
                    help='the size of d_a')
best_micro = 0


def main():
    global args, best_micro
    args = parser.parse_args()
    base1_corr, base2_corr, embed = data_got.corrset(batch_size=args.batch_size,sample_num=args.num_sample,num_class=args.num_class)
    ctail_loader= data_got.Tcorr_data(batch_size=args.batch_size,tsample_num=args.tnum_sample,num_class=args.num_class)
    oldtail_loader = data_got.oldtail_loader(batch_size=args.batch_size, num_class=args.num_class)
    test_loader,_ = data_got.test_loader(batch_size=args.batch_size)
    embed = torch.from_numpy(embed).float()
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    model = models.Net(args.batch_size, args.lstm_hid_dim, args.d_a, 54, embed).cuda()
    print('==> Reading from model checkpoint..')
    assert os.path.isfile(args.model), 'Error: no model checkpoint directory found!'
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model checkpoint '{}' (epoch {})"
          .format(args.model, checkpoint['epoch']))
    cudnn.benchmark = True
    Q=torch.from_numpy(np.load("v.npy")).cuda()
    x_tail,y_tail,rep_y = corr_getfunc(base1_corr, base2_corr,ctail_loader,oldtail_loader,model)
    x_tail=x_tail.cpu()
    y_tail=y_tail.cpu()
    # model.classifier.fc = nn.Linear(128, 34, bias=False)
    tail = data_utils.TensorDataset(x_tail,rep_y)
    tail_loader = data_utils.DataLoader(tail, 1200, shuffle=True,drop_last=True)
    # head_weight =model.classifier.fc.weight.data
    trans_model = models.Transfer().cuda()
    # step1: xiu gai feature close to few-shot p tail
    model_criterion = nn.MSELoss()
    model_optimizer = torch.optim.Adam(trans_model.parameters(), lr=0.01, betas=(0.9, 0.99))




    for epoch in range(10):
        train_loss= train(tail_loader, trans_model, model_criterion, model_optimizer,Q)
        print("train loss",train_loss)
    output_all= test(tail_loader,trans_model,Q)

    new_tail = data_utils.TensorDataset(output_all, y_tail)
    tail_aug = data_utils.DataLoader(new_tail, args.batch_size, shuffle=True, drop_last=True)
    #use new tail to rebuild tail classifier
    rebuild (tail_aug, model)
    final_test(test_loader,model)


def corr_getfunc(train_loader1,train_loader2,ctail_loader,oldtail_loader, model):
    base1_rep=[]
    base2_rep=[]
    tail_rep=[]
    return gen_tail,gen_y,rep_y


def train(train_loader, trans_model, criterion, optimizer,Q):
    losses = AverageMeter()
    trans_model.train()
    # model.classifier.fc = nn.Linear(128, 34, bias=False)
    for batch_idx, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        output = trans_model(input)
        loss1 = criterion(output, target)


        dis=torch.mm(Q.t(),Q)
        output2=torch.mm(output,dis)
        loss2 = criterion(output2, output)


        output3 = torch.mm(output,Q)
        avg = torch.sum(output3, 0)/1200
        target3=avg.expand(output3.shape)
        loss3=-0.01*criterion(output3,target3)
        # print("sim loss1",loss1)
        # print("dis loss2",loss2)
        # print("diversity loss3",loss3)
        loss=loss1+loss2+loss3
        losses.update(loss.item(), input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses.avg

def test(train_loader, trans_model,Q):
    output_all=[]
    print("this is add distribution")
    for batch_idx, (input, target) in enumerate(train_loader):
        input = input.cuda()
        # target = target.cuda()
        output = trans_model(input)
        # dis = torch.mm(Q.t(), Q)
        # output = torch.mm(output, dis)
        output_all.extend(output)
    output_all=torch.stack(output_all)
    return output_all

def rebuild(val_loader, model):
    print("**************************************")
    print("**************************************")
    print("Data enhancement phase")
    score_micro = np.zeros(3)
    test_p1, test_p3, test_p5 = 0, 0, 0
    test_ndcg1, test_ndcg3, test_ndcg5 = 0, 0, 0
    model.train()
    for batch_idx, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        output = model.classifier(input)
        target = target.data.cpu().float()
        output = output.data.cpu()
        _p1, _p3, _p5 = precision_k(output.topk(k=5)[1].numpy(), target.numpy(), k=[1, 3, 5])
        test_p1 += _p1
        test_p3 += _p3
        test_p5 += _p5

        _ndcg1, _ndcg3, _ndcg5 = ndcg_k(output.topk(k=5)[1].numpy(), target.numpy(), k=[1, 3, 5])
        test_ndcg1 += _ndcg1
        test_ndcg3 += _ndcg3
        test_ndcg5 += _ndcg5
        output[output > 0.5] = 1
        output[output <= 0.5] = 0
        # for l in range(54):
        #     F1[l] += f1_score(target[:, l], output[:, l], average='binary')
        #     precision[l] += precision_score(target[:, l], output[:, l], average='binary')
        #     recall[l] += recall_score(target[:, l], output[:, l], average='binary')
        score_micro += [precision_score(target, output, average='micro'),
                        recall_score(target, output, average='micro'),
                        f1_score(target, output, average='micro')]
    np.set_printoptions(formatter={'float': '{: 0.3}'.format})
    print('the result of micro: \n', score_micro / len(val_loader))
    test_p1 /= len(val_loader)
    test_p3 /= len(val_loader)
    test_p5 /= len(val_loader)

    test_ndcg1 /= len(val_loader)
    test_ndcg3 /= len(val_loader)
    test_ndcg5 /= len(val_loader)

    print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (test_p1, test_p3, test_p5))
    print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (test_ndcg1, test_ndcg3, test_ndcg5))
    # print('the result of F1: \n', F1 / len(val_loader))
    # print('the result of precision: \n', precision / len(val_loader))
    # print('the result of recall: \n', recall / len(val_loader))

    return score_micro / len(val_loader)


def final_test(val_loader, model):
    print("**************************************")
    print("**************************************")
    print("this is the final results")
    losses = AverageMeter()
    score_micro = np.zeros(3)
    test_p1, test_p3, test_p5 = 0, 0, 0
    test_ndcg1, test_ndcg3, test_ndcg5 = 0, 0, 0
    precision = np.zeros(54)
    recall = np.zeros(54)
    F1 = np.zeros(54)

    # model.eval()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            target = target.data.cpu().float()
            output = output.data.cpu()
            _p1, _p3, _p5 = precision_k(output.topk(k=5)[1].numpy(), target.numpy(), k=[1, 3, 5])
            test_p1 += _p1
            test_p3 += _p3
            test_p5 += _p5

            _ndcg1, _ndcg3, _ndcg5 = ndcg_k(output.topk(k=5)[1].numpy(), target.numpy(), k=[1, 3, 5])
            test_ndcg1 += _ndcg1
            test_ndcg3 += _ndcg3
            test_ndcg5 += _ndcg5
            output[output > 0.5] = 1
            output[output <= 0.5] = 0
            for l in range(54):
                F1[l] += f1_score(target[:, l], output[:, l], average='binary')
                precision[l] += precision_score(target[:, l], output[:, l], average='binary')
                recall[l] += recall_score(target[:, l], output[:, l], average='binary')
            score_micro += [precision_score(target, output, average='micro'),
                            recall_score(target, output, average='micro'),
                            f1_score(target, output, average='micro')]
        np.set_printoptions(formatter={'float': '{: 0.3}'.format})
        print('the result of micro: \n', score_micro / len(val_loader))
        test_p1 /= len(val_loader)
        test_p3 /= len(val_loader)
        test_p5 /= len(val_loader)

        test_ndcg1 /= len(val_loader)
        test_ndcg3 /= len(val_loader)
        test_ndcg5 /= len(val_loader)

        print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (test_p1, test_p3, test_p5))
        print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (test_ndcg1, test_ndcg3, test_ndcg5))
        print('the result of F1: \n', F1 / len(val_loader))
        print('the result of precision: \n', precision / len(val_loader))
        print('the result of recall: \n', recall / len(val_loader))

        return score_micro / len(val_loader)
if __name__ == '__main__':
    main()