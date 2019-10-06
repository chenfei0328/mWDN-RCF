import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
from torchvision import transforms, utils
from tqdm import tqdm
from model import *
from load_data import read_data

EPOCH = 1000
BATCH_SIZE = 16
LEARNING_RATE = 0.01
alpha , beta = 0.1 , 0.1

class MyDataSet(Dataset):
    def __init__(self, x_train, y_train, normalize=True):
        self.x_train = x_train
        self.y_train = y_train
        self.normalize = normalize
        if self.normalize:
            self.mean = np.mean(self.x_train)
            self.std = np.std(self.x_train)

    def __getitem__(self, item):
        ts = self.x_train[item]
        label = int(self.y_train[item])
        if self.normalize:
            ts = (ts - self.mean) / self.std
        return ts, label

    def __len__(self):
        return len(self.x_train)

def loss_function(output, label):
    num_classes = len(output[0])
    output = torch.t(output)
    final_loss = 0.0
    for l in label:
        k = 0
        one_hot = torch.zeros([1, num_classes])
        one_hot[0, l - 1] = 1
        one_hot_inv = torch.ones([1, num_classes]) - one_hot
        loss = torch.mm(one_hot, torch.log(output[:, k].view(-1, 1))) + torch.mm(one_hot_inv, torch.log(torch.ones([num_classes, 1]) - output[:, k].view(-1, 1)))
        final_loss += loss
        k += 1
    #print(final_loss)
    return final_loss

def train():
    # 后续从x_train抽出30%作为验证集
    x_train, y_train, x_test, y_test = read_data()

    input_size = len(x_train[0])
    class_num = len(np.unique(y_train))
    #print(input_size)
    #print(output_size)

    train_set = MyDataSet(x_train, y_train, normalize=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    test_set = MyDataSet(x_test, y_test, normalize=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # for i, batch in enumerate(train_loader):
    #     print(i)
    #     # batch[0]为数据,batch[1]为标签
    #     print(batch[0], batch[1])
    #     break

    nn.CrossEntropyLoss()
    net = mWDN_RCF(input_size, class_num)
    #print(net)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    if torch.cuda.is_available():
        net = net.cuda()

    train_losses = []
    for e in tqdm(range(EPOCH)):
        print('EPOCH {e} of {E}'.format(e=e+1, E=EPOCH))
        train_loss = 0
        for i, batch in enumerate(train_loader):
            #print(i)
            input_data = batch[0].view(BATCH_SIZE, -1)
            #print(input_data.shape)
            #print(input_data.shape)
            c1, c2, c3 = net(input_data)
            # print(c1.shape) torch.Size([16, 1])
            # print(batch[1].shape) torch.Size([16])
            loss = (1 * loss_function(c1, batch[1]) + 2 * loss_function(c2, batch[1]) + 3 * loss_function(c3, batch[1])) / 3 / BATCH_SIZE
            print(loss)

            W_mWDN1_H = net.mWDN1_H.weight.data
            W_mWDN1_L = net.mWDN1_L.weight.data
            W_mWDN2_H = net.mWDN2_H.weight.data
            W_mWDN2_L = net.mWDN2_L.weight.data
            W_mWDN3_H = net.mWDN3_H.weight.data
            W_mWDN3_L = net.mWDN3_L.weight.data
            # print(W_mWDN3_L.shape)
            # print(W_mWDN3_L.dtype)
            # print(net.cmp_mWDN3_L.shape)
            # print(net.cmp_mWDN3_L.dtype)
            # print(torch.norm((W_mWDN1_L - net.cmp_mWDN1_L), 2))
            L_loss = torch.norm((W_mWDN1_L - net.cmp_mWDN1_L), 2) + torch.norm((W_mWDN2_L - net.cmp_mWDN2_L), 2) + torch.norm((W_mWDN3_L - net.cmp_mWDN3_L), 2)
            H_loss = torch.norm((W_mWDN1_H - net.cmp_mWDN1_H), 2) + torch.norm((W_mWDN2_H - net.cmp_mWDN2_H), 2) + torch.norm((W_mWDN3_H - net.cmp_mWDN3_H), 2)
            loss += alpha * L_loss + beta * H_loss

            train_loss += loss * BATCH_SIZE

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

    final_loss = train_losses[-1]

if __name__ == '__main__':
    train()