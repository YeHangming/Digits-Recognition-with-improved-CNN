from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import random
import numpy as np
train_loss_epoch = []
test_loss_epoch = []
test_acc_epoch = []

class DCNN(nn.Module):
    def __init__(self):
        super(DCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2304, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    count = 0
    train_loss=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        train_loss+=loss.item()
        count+=1
        # Calculate gradients
        loss.backward()
        # Optimize the parameters according to the calculated gradients
        optimizer.step()
        '''
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        '''
    train_loss_epoch.append(train_loss/count)


def test(model, device, test_loader,label,draw):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx,(data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_loss_epoch.append(test_loss)
    test_acc_epoch.append(100. * correct / len(test_loader.dataset))
    '''
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    '''

    if draw:
        result = pred.eq(target.view_as(pred)).tolist()
        for (idx, item) in enumerate(result):
            if (item == [label]):
                pic = data[idx, 0, :, :]
                prob = F.softmax(output[idx, :], dim=0, dtype=float)

                plt.imshow(pic.cpu(), cmap='gray')
                plt.title("target:" + str(target.tolist()[idx]) + "  actual:" + str(pred.tolist()[idx][0]))
                filename = str(['{:.4f}'.format(i) for i in prob.tolist()])
                plt.savefig(str(label) + '/' + str(idx) + filename + ".jpg")
                plt.show()


    return 100. * correct / len(test_loader.dataset)


def tuning(arg_name, arg_value):
    result = {}
    plt.figure()
    for i in arg_value:
        acc = main(['--' + arg_name, i])
        # result[i] = [list(train_loss_epoch),list(test_loss_epoch),list(test_acc_epoch)]
        plt.plot(list(range(1, 16)), list(test_acc_epoch), label=arg_name + ': ' + i)
        print(acc)
        train_loss_epoch.clear()
        test_loss_epoch.clear()
        test_acc_epoch.clear()

    plt.legend()
    plt.xlabel('Epoch times')
    plt.ylabel('Accuracy')
    plt.show()

def random_search():
    MAX_EVALS = 64
    param_grid = {
        '--batch-size': [2 ** i for i in range(5, 9)],
        '--lr': list(np.logspace(np.log10(0.05), np.log10(5), base=10, num=50)),
        '--epochs': [10, 15, 20],
        '--gamma': list(np.arange(0.8, 1.0, 0.5))
    }
    best_score = 0
    best_param = {}

    for i in range(MAX_EVALS):
        random.seed(i)
        para = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        para_list = []
        for i in para.keys():
            para_list.append(i)
            para_list.append(str(para[i]))
        score = main(para_list)
        if score > best_score:
            best_para = para.copy()
            best_score = score
            print(para, score)
    return best_para


def main(aug,improved_net):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    # batch_size is a crucial hyper-parameter
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        # Adjust num worker and pin memory according to your computer performance
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Normalize the input (black and white image)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    # Make train dataset split and augment
    if aug:
        train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(5, 5)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        dataset1 = datasets.MNIST('./data', train=True, download=True,
                                  transform=train_transform)
    else:
        dataset1 = datasets.MNIST('./data', train=True, download=True,
                                  transform=transform)


    # Make test dataset split
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)

    # Convert the dataset to dataloader, including train_kwargs and test_kwargs
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Put the model on the GPU or CPU
    if improved_net:
        model = DCNN().to(device)
    else:
        model=Net().to(device)

    # Create optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Create a schedule for the optimizer
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Begin training and testing
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader,False,False)
        scheduler.step()

    # Save the model
    if args.save_model:
        torch.save(model.state_dict(), "improved_mnist_cnn.pt")
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':

    print(main(False,False))
    plt.plot(list(range(1, 16)), list(test_acc_epoch), label='default model')
    test_acc_epoch.clear()

    print(main(True,True))
    plt.plot(list(range(1, 16)), list(test_acc_epoch), label='improved model')
    test_acc_epoch.clear()

    plt.legend()
    plt.title('Comparison between default and improved model')
    plt.savefig('Comparison.jpg')
    plt.show()
