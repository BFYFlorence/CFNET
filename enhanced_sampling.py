import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data
import my.Lacomplex as lc

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.optim as optim
from torch.optim import lr_scheduler
import os
from torch.functional import F
import seaborn as sns

torch.manual_seed(1997315)  # random seed
np.set_printoptions(threshold=10)
torch.set_printoptions(threshold=sys.maxsize, sci_mode=False)
# torch.set_printoptions(profile="full")

# ~:条件取反

class discriminator(nn.Module):
    def __init__(self, input_s, output_s):
        super(discriminator, self).__init__()
        # self.con1 = nn.Conv2d(2, 3, (2,2))  # Convolution kernel number, depth, (convolution kernel size)

    def forward(self, x):
        # return self.con1(x)
        pass

class generator(nn.Module):
    def __init__(self, input_s, output_s):
        super(generator, self).__init__()

    def forward(self, x):
        pass


class dataset(data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        record, label = self.data[index], self.labels[index]
        return record, label

    def __len__(self):
        return len(self.data)


def train_GAN():
    load_data = np.load("./N-CA_reduced.npy", allow_pickle=True)
    data_pool = dataset(load_data, Variable(torch.ones([len(load_data), 1])))
    # (len,1) to make sure index is not out of boundary, or else (1,len) may raise
    # an exception from __getitem__ because in this case it needs 2 indexes instead of 1
    loader = data.DataLoader(
        dataset=data_pool,
        batch_size=300,
        shuffle=True,
        num_workers=0,  # whether read data using MPI
    )
    for epoch in range(1):  # determine the number of loops over whole data
        for step, (batch_x, batch_y) in enumerate(loader):  # the remaining will not be dropped
            print("Epoch: ", epoch, "| Step: ", step, '\n',
                  "batch x: ", '\n', batch_x.numpy(), '\n',
                  "batch y: ", '\n', batch_y.numpy().T)


# print(nobind[13249])
# print(nobind.shape)

"""
input = np.random.normal(loc=0, scale=1, size=(150, 3))
print(input.shape)

kernel = torch.tensor(kernel).unsqueeze(0).unsqueeze(0)

weight = nn.Parameter(data=kernel, requires_grad=False)

test = torch.tensor(test).unsqueeze(1)

test = test.float()
print(test.size())"""

class MyLoss(nn.Module):
    # implementation of classification tasks scaled to 0,1
    # when design your own loss function, make sure that all key operations are done by torch
    # so that the grad_info could be passed
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, x: torch.tensor,  # 300,1
                y: torch.tensor,  # 300, 2
                ):
        x = (x + 1) / 2
        x = torch.hstack((1 - x, x))  # 300,2
        # x = torch.exp(x)  # 300,2
        # sum = torch.sum(x, dim=1, keepdim=True)
        # x = torch.div(x, sum)

        # Taking out elements by indexing and customizing a kernel to perform matrix multiplication have similar effects,
        # so the gradient can be passed on
        # weight = model.state_dict()["map3.weight"].shape  # (1,16)
        loss = torch.mean(-torch.log(x[torch.arange(x.shape[0]), y.argmax(axis=1)]))
        return loss

class BiClassifier(nn.Module):
    def __init__(self, input_s, output_s):
        super(BiClassifier, self).__init__()
        self.map1 = nn.Linear(input_s, 10)
        self.map2 = nn.Linear(10, 10)
        self.map3 = nn.Linear(10, output_s)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.map1(x))
        x = self.tanh(self.map2(x))
        x = self.tanh(self.map3(x))
        return x

def accuracy(classification_result, labels):
    classification_result = (classification_result + 1) / 2
    classification_result = torch.round(classification_result)
    classification_result = torch.squeeze(classification_result)
    return torch.sum(torch.eq(classification_result, labels.argmax(1))) / labels.size()[0]

def save_checkpoint(state, is_best, filename='./checkpoint.pth.tar'):
    # Save checkpoint if a new best is achieved
    if is_best:
        print("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print("=> Validation Accuracy did not improve")

def train_classifier():
    # load data
    dir = "./dis100dih200_64_16_1/"
    bind = np.load("./bind.npy", allow_pickle=True)  # (15000, 2476)
    nobind = np.load("./nobind.npy", allow_pickle=True)  # (15000, 2476)
    # extract feat
    # attention! plumed use 'nm' as the standard unit, therefore you need to scale the distance metrics
    indice = np.load("./dis100dih200_64_16_1/dihedral_n_weightIndice_order.npy", allow_pickle=True)  # (1, 200)

    modi_bind = np.hstack((bind[:, :340]/10, bind[:, np.squeeze(340 + indice)]))  # (15000, 540)
    modi_nobind = np.hstack((nobind[:, :340]/10, nobind[:, np.squeeze(340 + indice)]))  # (15000, 540)

    # data shape
    input_size = modi_bind.shape[1]  # 540
    output_size = 1
    epochs = 5
    batch_size = 300
    sample_size = modi_bind.shape[0] * 2  # 30000
    train_size = int(sample_size * .8)  # 24000
    test_size = int(sample_size * .2)  # 6000
    acc = 0
    loss = 0
    lowest_loss = 9999
    checkpoint_path = './{0}'.format(dir)
    # launchTimestamp = os.times()

    labels = torch.tensor(  # torch.Size([30000, 2])
        np.concatenate((np.hstack((np.zeros(shape=(modi_bind.shape[0], 1)), np.ones(shape=(modi_bind.shape[0], 1)))),
                        np.hstack((np.ones(shape=(modi_nobind.shape[0], 1)),
                                   np.zeros(shape=(modi_nobind.shape[0], 1)))))))  # 1111100000

    load_data = torch.tensor(np.concatenate((modi_bind, modi_nobind), axis=0))  # torch.Size([30000, 540])
    data_pool = dataset(load_data.float(), labels.float())
    train_set, test_set = data.random_split(data_pool, [train_size, test_size])

    train_loader = data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # whether read data using MPI
    )

    test_loader = data.DataLoader(
        dataset=test_set,
        batch_size=test_size,
        shuffle=True,
        num_workers=0,  # whether read data using MPI
    )

    # initialize model
    classifier = BiClassifier(input_s=input_size, output_s=output_size)

    # initialize optimizer and loss function
    learning_rate = 0.0001
    cost = MyLoss()  # Binary cross entropy
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.00001)

    for epoch in range(epochs):  # determine the number of loops over whole data
        # torch.Size([300, 540]) torch.Size([300, 2])
        for step, (train_x, train_y) in enumerate(train_loader):  # the remaining will not be dropped
            classifier.zero_grad()
            classification_result = classifier(train_x)  # (300,1)
            print(classification_result)
            # result = np.concatenate(())

            loss = cost(classification_result, train_y, classifier)
            loss.backward()
            optimizer.step()  # update parameters after calling backward
            for step_test, (test_x, test_y) in enumerate(test_loader):
                acc = accuracy(classifier(test_x), test_y)
            print(" ===============================\n",
                  "${:^6}|{:^21d}||\n".format("epochs", epoch),
                  "-------------------------------\n",
                  "${:^6}|{:^21d}||\n".format("step", step),
                  "-------------------------------\n",
                  "${:^6}|{:^21f}||\n".format("loss", loss),
                  "-------------------------------\n",
                  "${:^6}|{:^21f}||\n".format("acc", acc),
                  "===============================\n")

        if loss.data < lowest_loss:
            lowest_loss = loss
            torch.save({'epoch': epoch + 1,
                        'state_dict': classifier.state_dict(),
                        'best_loss': lowest_loss,
                        'optimizer': optimizer.state_dict()},
                        checkpoint_path + 'm-' + str("%.4f" % lowest_loss) + '.pth.tar')
            torch.save(classifier, './{0}/model_test.pt'.format(dir))
        print(optimizer.state_dict()['param_groups'])
        lr_scheduler.step()  # update learning rate

# train_classifier()
class NN_analysis:
    def __init__(self):
        pass
    def test_model(self, model):
        dir = "dis340dih200_3"
        model_name = "m-0.0226.pth.tar"
        model_CKPT = torch.load("{0}/{1}".format(dir, model_name), map_location=torch.device('cpu'))
        model.load_state_dict(model_CKPT['state_dict'])
        model.double()

        bind = np.load("./test_dataset/reverse_test.npy", allow_pickle=True)
        nobind = np.load("./nobind.npy", allow_pickle=True)

        # extract feat
        indice = np.load("./{0}/dihedral_n_weightIndice_order.npy".format(dir), allow_pickle=True)
        modi_bind = np.hstack((bind[:, :340]/10., bind[:, np.squeeze(340 + indice)]))
        modi_nobind = np.hstack((nobind[:, :340]/10., nobind[:, np.squeeze(340 + indice)]))
        """labels = torch.tensor(
            np.concatenate((np.hstack((np.zeros(shape=(modi_bind.shape[0], 1)), np.ones(shape=(modi_bind.shape[0], 1)))),
                            np.hstack((np.ones(shape=(modi_nobind.shape[0], 1)),
                                       np.zeros(shape=(modi_nobind.shape[0], 1)))))))  # 1111100000"""
        # load_data = torch.tensor(np.concatenate((modi_bind, modi_nobind), axis=0))
        load_data = torch.tensor(modi_bind)
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)

        result = model(load_data)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('', fontsize=20)
        ax1.set_xlabel('feat_num', fontsize=20)
        ax1.set_ylabel('weights', fontsize=20)
        ax1.scatter(range(result.shape[0]), result.detach().numpy(), s=.1)  # 12 color

        """data_pool = dataset(load_data.float(), labels.float())
        test_loader = data.DataLoader(
            dataset=data_pool,
            batch_size=15000,
            shuffle=True,
            num_workers=0,  # whether read data using MPI
        )
        
        for step, (train_x, train_y) in enumerate(test_loader):
            result = model(train_x)
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.set_title('', fontsize=20)
            ax1.set_xlabel('feat_num', fontsize=20)
            ax1.set_ylabel('weights', fontsize=20)
            ax1.scatter(range(result.shape[0]), result.detach().numpy(), s=.1)"""


        plt.show()

    def output_parameters(self, model_CKPT, dir):
        keys = list(model_CKPT['state_dict'].keys())

        for key in range(len(keys)):
            items = model_CKPT['state_dict'][keys[key]]
            print(items.size())
            if len(items.size()) == 2:
                name = "./{2}/{0}_{1}_WEIGHTS_{3}.dat".format(items.size()[0], items.size()[1], dir, key)
                with open(name, 'a') as file_w:
                    for i in range(items.size()[0]):
                        for j in range(items.size()[1]):
                            file_w.write(',')
                            file_w.write(str(items[i][j].tolist()))
                file_w.close()
            elif len(items.size()) == 1:
                name = "./{1}/{0}_BIAS_{2}.dat".format(items.size()[0], dir, key)
                with open(name, 'a') as file:
                    for i in range(items.size()[0]):
                        file.write(',')
                        file.write(str(items[i].tolist()))
                file.close()

    def test_model_single(self, model, lc):
        # Since there will be errors when exporting the coordinates,
        # the dihedral angle value calculated by myself will be slightly different from that calculated by plumed,
        # but this is normal
        with open("./forward/COLVAR", 'r') as file:
            calculation = []
            for i in file.readlines():
                record = i.strip()
                if record[0] != '#':
                    calculation.append(record)
                else:
                    ARG = record.split()[3:]
        file.close()

        dir = "dis340dih200_3"
        model_name = "m-0.0226.pth.tar"
        model_CKPT = torch.load("{0}/{1}".format(dir, model_name), map_location=torch.device('cpu'))
        model.load_state_dict(model_CKPT['state_dict'])
        model = model.double()

        target_pdb = 50000
        dis = lc.single_LDA_Dis("/Users/erik/PycharmProjects/Lacomplex/forward/md{0}_pbc_recover.pdb".format(target_pdb), "dis")
        dih = lc.single_LDA_dih("/Users/erik/PycharmProjects/Lacomplex/forward/md{0}_pbc_recover.pdb".format(target_pdb))
        data = torch.tensor(np.hstack((dis / 10., dih)), requires_grad=True)

        nn_out = model(data)
        print(nn_out)


        # first element is time
"""        for k in range(541):
            if torch.abs(float(calculation[j].split()[k+1]) - all[0][k].data) >0.001:
                print("large deviation detected: ",
                      calculation[j].split()[k+1],
                      all[0][k].data,
                      k)
                print(ARG[k])"""


        # dih_indice = np.load("./dihedral_n_weightIndice_order.npy", allow_pickle=True)
        # print(dih_indice.shape)
        # print(dih_indice[0][186])

        # print(model.state_dict()["map3.weight"].shape)

class AMSoftmax(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes=10,
                 m=0.35,
                 s=30):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        # l2 norm
        x_norm = torch.norm(x, dim=1, keepdim=True).clamp(min=1e-12)  # Clamp to a certain interval
        x_norm = torch.div(x, x_norm)
        print("x_norm:", x_norm, x_norm.shape)
        w_norm = torch.norm(self.W, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        print("w_norm:", w_norm, w_norm.shape)
        costh = torch.mm(x_norm, w_norm)  # Matrix multiplication
        # print(x_norm.shape, w_norm.shape, costh.shape)
        lb_view = lb.view(-1, 1)  # n行1列
        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)  # in-place edit
        # dim=1
        # self[i][index[i][j]] = src[i][j]
        print(delt_costh.shape)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss, costh_m_s

NN_analysis = NN_analysis()
lc = lc.Lacomplex()

"""if __name__ == '__main__':
    criteria = AMSoftmax(20, 5)
    a = torch.randn(10, 20)  # Returns a tensor with a mean of 0 and a variance of 1
    print("a:", a, a.shape)
    lb = torch.randint(0, 5, (10, ), dtype=torch.long)  # Returns 0-4, a tensor of shape (10,1)
    print("lb:", lb, lb.shape)

    loss,_ = criteria(a, lb)
    loss.backward()"""

    # print(loss.detach().numpy())
    # print(list(criteria.parameters())[0].shape)
    # print(type(next(criteria.parameters())))

NN_analysis.test_model_single(BiClassifier(input_s=540, output_s=1), lc)

# NN_analysis.test_model(BiClassifier(input_s=540, output_s=1))

# model_CKPT = torch.load("./dis340dih200_3/m-0.0226.pth.tar", map_location=torch.device('cpu'))
# NN_analysis.output_parameters(model_CKPT, "dis340dih200_3")

