import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ParamTracker:
    def __init__(self, params):
        self.fractal_steps = 12
        self.params = []
        self.init_params = []
        # self.epoch_params = []
        for tensor in params:
            self.init_params.append(tensor.clone())

        for _ in range(self.fractal_steps):
            params_copy = []
            for tensor in self.init_params:
                params_copy.append(tensor.clone())
            self.params.append(params_copy)
            # self.epoch_params.append(tensor.clone())
        self.step_count = 0
        self.dist = []
        self.cum_dists = []
        for _ in range(self.fractal_steps):
            self.dist.append(0.)
            self.cum_dists.append([])

    def step(self, new_params):
        new_params = [tensor for tensor in new_params]
        self.step_count += 1
        with torch.no_grad():
            for fstep in range(self.fractal_steps):
                if self.step_count % (1<<fstep) != 0:
                    continue
                # print("{} {}".format(fstep, len(self.params)))
                # print("{}".format(len(self.init_params)))
                total = 0.
                for (i, tensor) in enumerate(new_params):
                    total += torch.sum(self.params[fstep][i] * tensor).item()
                    self.params[fstep][i] = tensor.clone()
                dist = np.sqrt(total)
                self.dist[fstep] += dist
                self.cum_dists[fstep].append(self.dist[fstep])

    def get_dist(params1, params2):
        with torch.no_grad():
            total = 0.
            for (i, tensor) in enumerate(params1):
                total += torch.sum(params2[i] * tensor).item()
            dist = np.sqrt(total)
            return dist

    # def cumulative_dist(self):
    #     return self.dist

    # def save_epoch(self):
    #     for (i, tensor) in enumerate(self.params):
    #         self.epoch_params[i] = tensor

    # def epoch_dist(self):
    #     return ParamTracker.get_dist(self.epoch_params, self.params)

    # def init_to_last_dist(self):
    #     return ParamTracker.get_dist(self.init_params, self.params)

def train():
    pass

def test(loader, model, criterion):
    model.eval()
    test_loss = 0.
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            test_loss += criterion(outputs, labels)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
        test_loss /= len(loader.dataset)
        accuracy = correct / len(loader.dataset)
        print("\nTest set: Average loss {}, accuracy: {}/{} ({}%)".format(
            test_loss, correct, len(loader.dataset), 100.*accuracy
        ))

net = Net()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

param_tracker = ParamTracker(net.parameters())

prev_cum_dist = 0.
for epoch in range(40):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        param_tracker.step(net.parameters())

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

    test(testloader, net, criterion)
    # ed = param_tracker.epoch_dist()
    # ecd = param_tracker.cumulative_dist() - prev_cum_dist
    # prev_cum_dist = ecd
    # param_tracker.save_epoch()
    # print("ecd: {} ed: {} ratio: {}".format(ecd, ed, ecd/ed))

print('Finished Training')
# print("cumulative dist: {}".format(param_tracker.cumulative_dist()))
# print("dist: {}".format(param_tracker.init_to_last_dist()))

fig, ax = plt.subplots()

# for (fdim, dists) in enumerate(param_tracker.cum_dists):
#     x_coords = [i * 1<<fdim for i in range(len(dists))]
#     ax.plot(x_coords, dists)

# calculate fractal dimensions

for fdim in range(1, len(param_tracker.cum_dists)):
    # (sample rate 1/sample rate 2)^n = (length 1/length 2)
    # where the dimension is 1 + n
    # thus n = log(length 1/length 2)/log(sample rate 1/sample rate 2)
    dists = param_tracker.cum_dists[fdim]
    x_coords = [i * 1<<fdim for i in range(len(dists))]
    y_coords = []
    for i in range(len(param_tracker.cum_dists[fdim])):
        len1 = param_tracker.cum_dists[fdim][i]
        len2 = param_tracker.cum_dists[0][(1<<fdim)*i]
        y_coords.append(1.+(np.log(len2/len1)/np.log(1<<fdim)))

    ax.plot(x_coords, y_coords)
    

ax.set_xlabel('steps')
ax.set_xlabel('cumulative distance')
plt.show()
