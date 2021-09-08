import numpy as np
from mxnet.contrib import text
import torch.utils.data as data_utils
import torch
import random
# import seaborn as sns
import scipy.sparse as sp


def Bload_data(batch_size=60):


    X_trn = np.load("data/AAPD/X_tra.npy")
    Y_trn = np.load("data/AAPD/y_trn.npy")
    X_tst = np.load("data/2AAPD/X_val.npy")
    Y_tst = np.load("data/2AAPD/y_val.npy")

    embed = text.embedding.CustomEmbedding('data/AAPD/word_embed.txt')

    base_train = data_utils.TensorDataset(torch.from_numpy(X_trn).type(torch.LongTensor),
                                          torch.from_numpy(Y_trn).type(torch.LongTensor))

    base_val = data_utils.TensorDataset(torch.from_numpy(X_tst).type(torch.LongTensor),
                                          torch.from_numpy(Y_tst).type(torch.LongTensor))


    Btrain_loader = data_utils.DataLoader(base_train, batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
    Btest_loader = data_utils.DataLoader(base_val, batch_size, shuffle=False,drop_last=True, num_workers=4, pin_memory=True)
    return Btrain_loader, Btest_loader, embed.idx_to_vec.asnumpy()



def joint_loader(batch_size=60):
    X_trn = np.load("data/2AAPD/X_tra.npy")
    Y_trn = np.load("data/2AAPD/y_trn.npy")
    X_tst = np.load("data/2AAPD/X_val.npy")
    Y_tst = np.load("data/2AAPD/y_val.npy")

    embed = text.embedding.CustomEmbedding('data/AAPD/word_embed.txt')
    test_data = data_utils.TensorDataset(torch.from_numpy(X_tst).type(torch.LongTensor),
                                         torch.from_numpy(Y_tst).type(torch.LongTensor))

    base_train = data_utils.TensorDataset(torch.from_numpy( X_trn).type(torch.LongTensor),
                                          torch.from_numpy(Y_trn).type(torch.LongTensor))

    # base_val = data_utils.TensorDataset(torch.from_numpy(base_valX).type(torch.LongTensor),
    #                                       torch.from_numpy(base_valY).type(torch.LongTensor))


    Btrain_loader = data_utils.DataLoader(base_train, batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
    Btest_loader = data_utils.DataLoader(test_data, batch_size, shuffle=False,drop_last=True, num_workers=4, pin_memory=True)
    return Btrain_loader, Btest_loader, embed.idx_to_vec.asnumpy()

def Tcorr_data(batch_size=120,tsample_num=5,num_class=45):
    #AAPD data

    X_trn = np.load("data/AAPD/X_train.npy")
    Y_trn = np.load("data/AAPD/y_train.npy")
    novel1=Y_trn[...,num_class:]
    novel=novel1.T
    # novel_id=[]
    total = []
    etotal=[]
    for i in novel:
        novel_sam = []
        for m in range(5):
            w = np.nonzero(i)
            novel_sam.extend(random.sample(list(w[0]),tsample_num))
        total.append(novel_sam)
        # etotal.extend(novel_sam)
    novel_x=[]
    novel_y=[]
    for i in total:
        novel_x.extend(X_trn[i])
        novel_y.extend(novel1[i])
    novel_y=np.array(novel_y)
    novel_x=np.array(novel_x)

    ctail_data = data_utils.TensorDataset(torch.from_numpy(novel_x).type(torch.LongTensor),
                                         torch.from_numpy(novel_y).type(torch.LongTensor))
    ctail_loader = data_utils.DataLoader(ctail_data, batch_size, shuffle=False,drop_last=True, num_workers=4, pin_memory=True)
    return ctail_loader


def corrset(batch_size=30,sample_num=10,num_class=36):
    #AAPD data
    X_trn = np.load("data/AAPD/X_train.npy")
    Y_trn = np.load("data/AAPD/y_train.npy")
    embed = text.embedding.CustomEmbedding('data/AAPD/word_embed.txt')
    base = Y_trn[..., :num_class]

    Y_trn1 = base.T
    base2_sam = []
    base1_sam = []
    for i in Y_trn1:
        w = np.nonzero(i)
        base1_sam.append(random.sample(list(w[0]), sample_num))
        base2_sam.append(random.sample(list(w[0]), sample_num))
    base1_x = []
    base1_y = []
    base2_x = []
    base2_y = []
    for i in base1_sam:
        base1_x.extend(X_trn[i])
        base1_y.extend(Y_trn[i])
    base1_x = np.array(base1_x)
    base1_y = np.array(base1_y)
    for i in base2_sam:
        base2_x.extend(X_trn[i])
        base2_y.extend(Y_trn[i])
    base2_x = np.array(base2_x)
    base2_y = np.array(base2_y)
    corr1 = data_utils.TensorDataset(torch.from_numpy(base1_x).type(torch.LongTensor),
                                      torch.from_numpy(base1_y).type(torch.LongTensor))
    corr2 = data_utils.TensorDataset(torch.from_numpy(base2_x).type(torch.LongTensor),
                                      torch.from_numpy(base2_y).type(torch.LongTensor))
    base1_corr = data_utils.DataLoader(corr1, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    base2_corr = data_utils.DataLoader(corr2, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # return base1_corr,  embed.idx_to_vec.asnumpy()
    return base1_corr, base2_corr, embed.idx_to_vec.asnumpy()



def oldtail_loader(batch_size=120,num_class=45):
    #AAPD data
    X_trn = np.load("data/AAPD/X_train.npy")
    Y_trn = np.load("data/AAPD/y_train.npy")
    novel1=Y_trn[...,num_class:]
    novel=novel1.T
    novel_id = []
    for i in novel:
        m = np.nonzero(i)
        novel_id.extend(m[0])
    novel_id = list(np.unique(novel_id))
    novely_trn = torch.from_numpy(novel1[novel_id])
    novelx_trn = torch.from_numpy(X_trn[novel_id])
    oldtail_data = data_utils.TensorDataset(novelx_trn.type(torch.LongTensor),
                                         novely_trn.type(torch.LongTensor))

    oldtail_loader = data_utils.DataLoader(oldtail_data, batch_size, shuffle=False,drop_last=True, num_workers=4, pin_memory=True)

    return oldtail_loader


def test_loader(batch_size=80):
    #AAPD data
    embed = text.embedding.CustomEmbedding('data/AAPD/word_embed.txt')
    X_tst = np.load("data/AAPD/X_test.npy")
    Y_tst = np.load("data/AAPD/y_test.npy")





    test_data = data_utils.TensorDataset(torch.from_numpy(X_tst).type(torch.LongTensor),
                                         torch.from_numpy(Y_tst).type(torch.LongTensor))

    test_loader = data_utils.DataLoader(test_data,batch_size, shuffle=False, drop_last=True, pin_memory=True)
    return test_loader,embed.idx_to_vec.asnumpy()