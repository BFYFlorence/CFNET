import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import seaborn as sns


random_x = np.random.normal(loc=4, scale=1.25, size=128)
random_y = np.random.normal(loc=4, scale=1.25, size=128)
random_z = np.random.normal(loc=4, scale=1.25, size=128)

random_points = np.array([random_x,random_y,random_z]).T

# sample
sample_points_x = np.random.uniform(low=0, high=1,size=128)
sample_points_y = np.random.uniform(low=0, high=1,size=128)
sample_points_z = np.random.uniform(low=0, high=1,size=128)

sample_points = np.array([sample_points_x,sample_points_y,sample_points_z]).T
# print(sample_points.shape)

def get_moments(ds):
    finals = []
    for d in ds:
        mean = torch.mean(d)
        diffs = d - mean
        var = torch.mean(torch.pow(diffs, 2.0))
        std = torch.pow(var, 0.5)
        zscores = diffs/(std+0.0001)  # Laplace modification
        skews = torch.mean(torch.pow(zscores, 3.0))  # Skewness
        kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0  # Kurtosis, the kurtosis value of the data that completely obeys the normal distribution is 3
        final = torch.cat((mean.reshape(1, ), std.reshape(1, ), skews.reshape(1, ), kurtoses.reshape(1, )))
        finals.append(final)
    return torch.stack(finals)

def get_distribution_sampler(mu, sigma, batchSize, FeatureNum):
    return torch.tensor(np.random.normal(mu, sigma, (batchSize, FeatureNum)))

def get_generator_input_sampler(m, n):
    return torch.rand(m, n)  # Uniform Distribution

class generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.f = f

    def forward(self, x):
        x = self.relu(self.map1(x))
        x = self.relu(self.map2(x))
        x = self.map3(x)
        return x

class discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.f = f

    def forward(self, x):
        x = self.relu(self.map1(x))
        x = self.relu(self.map2(x))
        x = self.f(self.map3(x))
        return x

d_input_size = 4
d_hidden_size = 10
d_output_size = 1
discriminator_activation_function = torch.sigmoid

g_input_size = 50
g_hidden_size = 200
g_output_size = 500
generator_activation_function = torch.tanh

featureNum = g_output_size # A set of samples has 500 data that obey the normal distribution
minibatch_size = 10 # batch_size
num_epochs = 2001
d_steps = 20 # iterations of discriminator
g_steps = 20 # iterations of generator

D = discriminator(input_size=d_input_size,
                  hidden_size=d_hidden_size,
                  output_size=d_output_size,
                  f=discriminator_activation_function)

G = generator(input_size=g_input_size,
              hidden_size=g_hidden_size,
              output_size=g_output_size,
              f=generator_activation_function)

# ----------------------
# Initialize the optimizer and loss function
# ----------------------
d_learning_rate = 0.0001
g_learning_rate = 0.0001

criterion = nn.BCELoss()  # Binary cross entropy
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate)

d_exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max = d_steps*5, eta_min=0.00001)
g_exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max = g_steps*5, eta_min=0.00001)

G_mean = [] # The mean of the data generated by the generator
G_std = [] # The variance of the data generated by the generator

data_mean = 4
data_stddev = 1.25
batch_size = 1


for epoch in range(num_epochs):
    # -------------------
    # Train the Detective
    # -------------------
    for d_index in range(d_steps):
        # Train D on real+fake
        d_exp_lr_scheduler.step()  # Counter
        D.zero_grad()  # Cleaning gradient for every batch
        # Train D on real, label here is 1
        d_real_data = get_distribution_sampler(data_mean, data_stddev, minibatch_size, featureNum)  # ground truth
        d_real_decision = D(get_moments(d_real_data))  # Find the four important characteristics of the data
        d_real_error = criterion(d_real_decision, Variable(torch.ones([minibatch_size, 1])))  # calculate error
        d_real_error.backward()  # Backpropagation
        # Train D on fake, label here is 0
        d_gen_input = get_generator_input_sampler(minibatch_size, g_input_size)
        d_fake_data = G(d_gen_input)
        d_fake_decision = D(get_moments(d_fake_data))
        d_fake_error = criterion(d_fake_decision, Variable(torch.zeros([minibatch_size, 1])))
        d_fake_error.backward()
        # Optimizer
        d_optimizer.step()
    # -------------------
    # Train the Generator
    # -------------------
    for g_index in range(g_steps):
        # Train G on D's response(So that the x generated by G is judged to be 1)
        g_exp_lr_scheduler.step()
        G.zero_grad()
        gen_input = get_generator_input_sampler(minibatch_size, g_input_size)
        g_fake_data = G(gen_input)  # Make the generator generate samples
        dg_fake_decision = D(get_moments(g_fake_data))  # Judgment made by D
        g_error = criterion(dg_fake_decision, Variable(torch.ones([minibatch_size, 1])))
        G_mean.append(g_fake_data.mean().item())
        G_std.append(g_fake_data.std().item())
        g_error.backward()
        g_optimizer.step()
    if epoch % 10 == 0:
        print("Epoch: {}, G data's Mean: {}, G data's Std: {}".format(epoch, G_mean[-1], G_std[-1]))
        print("Epoch: {}, Real data's Mean: {}, Real data's Std: {}".format(epoch, d_real_data.mean().item(),
                                                                            d_real_data.std().item()))
        print('-' * 10)


