# ********************Import Modules-1
import os

from my_rbf import RBF
from distances import EuclideanDistances
from my_utils import *
import torch
from CFNET import CFnet
from preprocess import *
# import torch.autograd
import torch.utils.data as data
import torch.optim as optim
import random
import sys
print("Pytorch version: ", torch.__version__)
torch.set_printoptions(threshold=10, sci_mode=False)
# def setup_seed(seed):
#      torch.manual_seed(seed)
#      torch.cuda.manual_seed_all(seed)
#      np.random.seed(seed)
#      random.seed(seed)
#      torch.backends.cudnn.deterministic = True
# setup_seed(1997)

# ********************Configure-2

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['OMP_NUM_THREADS']='1'
device = torch.device('cuda:'+str(0))
# print(device)
torch.cuda.set_device(device)
print("Current device: ", torch.cuda.get_device_name(0))

# ********************Initialize molecules' properties and indices-3

Z_np = np.array([12, 12, 16, 12, 12, 12, 12, 12, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
batch_size = 1
nATOM = 19  # the number of atoms
epochs = 1000
Z_ts = torch.tile(torch.tensor(Z_np.ravel(), dtype=torch.int64),(batch_size,))

# --------------model initialization
nFM = 64  # the number of feature maps
cutoff = 30.  # rbf, the range of distances
gap = 0.1  # rbf, the interval of centers
n_embeddings = 100
n_interactions = 3
# ---------------

c7o2h10_md = np.load("./c7o2h10_md.npy", allow_pickle=True).item()
E_list = []
for key in c7o2h10_md.keys():
    E_list += c7o2h10_md[key]["E"]
E_ts = torch.tensor(np.array(E_list))

# Assignment grad
R_list = []
for key in c7o2h10_md.keys():
    R_list += c7o2h10_md[key]["cors"]
R_ts = torch.tensor(np.array(R_list), requires_grad=True)

R_ts = R_ts.cuda() # out-of-place
E_ts = E_ts.cuda()
Z_ts = Z_ts.cuda()

train_size = int(len(E_list) * 0.8)
test_size = int(len(E_list) * 0.2)

data_pool = dataset(R_ts.float(), E_ts.float())
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

# ********************Initialize CFNET-4
cfnet = CFnet(n_interactions=n_interactions, nFM=nFM, cutoff=cutoff, gap=gap, n_embeddings=n_embeddings, gpu=True).cuda()
optimizer = optim.Adam(cfnet.parameters(), lr=0.00001)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.00001)
dir = "train_model"
checkpoint_path = './{0}/'.format(dir)
if not os.path.exists(os.getcwd()+dir):
    os.system("mkdir ./{0}".format(dir))

def train():
    for epoch in range(epochs):
        for step, (R, E) in enumerate(train_loader):
            R = torch.squeeze(R)
            # keep grad
            R.retain_grad()
            mols = molecules(nATOM, Z_ts, R, batch_size)

            # idx_ik = seg_i
            idx_ik = mols["idx_ik"].cuda()  # [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]

            # idx_jk = idx_j  [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]
            idx_jk = mols["idx_jk"].cuda()
            # print("idx_jk:", idx_jk, idx_cfnetjk.shape)
            offset = mols["offset"].cuda()  # [20, 3]  [[0.,0.,0.,],...,[0.,0.,0.,]]

            # [0,...19]
            seg_j = mols["seg_j"].cuda()  # distance index
            seg_i_sum = mols["seg_i_sum"].cuda()  # [ 0,  4,  8, 12, 16]
            seg_m = mols["seg_m"].cuda()

            ratio_j = torch.tensor(1., dtype=torch.float32).cuda()
            rho = torch.tensor(.01, dtype=torch.float32).cuda()
            cfnet.zero_grad()
            Ep = cfnet(z=Z_ts, r=R, offsets=offset, idx_ik=idx_ik, idx_jk=idx_jk, idx_j=idx_jk, seg_m=seg_m, seg_i=idx_ik,
                       seg_j=seg_j, ratio_j=ratio_j, seg_i_sum=seg_i_sum)
            # Ep = Ep.requires_grad_(True)
            Ep.backward(retain_graph=True)
            Fp = -R.grad
            optimizer.zero_grad()

            loss, errors = CalLoss(Ep, Fp, E, None, rho, fit_forces=False)

            loss.backward()

            optimizer.step()
            print(" ===============================\n",
                  "${:^6}|{:^21d}||\n".format("epochs", epoch),
                  "-------------------------------\n",
                  "${:^6}|{:^21d}||\n".format("step", step),
                  "-------------------------------\n",
                  "${:^6}|{:^21f}||\n".format("loss", loss),
                  # "-------------------------------\n",
                  # "${:^6}|{:^21f}||\n".format("acc", acc),
                  "===============================\n")

            if step % 5000 == 0:
                lowest_loss = loss
                torch.save({'epoch': epoch + 1,
                            'state_dict': cfnet.state_dict(),
                            'best_loss': lowest_loss,
                            'optimizer': optimizer.state_dict()},
                            checkpoint_path + 'm-' + str("%.4f" % lowest_loss) + '.pth.tar')
                torch.save(cfnet, './{0}/model_test.pt'.format(dir))

        lr_scheduler.step()  # update learning rate