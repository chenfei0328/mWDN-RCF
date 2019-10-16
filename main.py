import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import *
from load_data import read_data, redefine_category, transform_category

EPOCH = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
alpha , beta = 0.01 , 0.01

def plot_loss(loss):
    plt.plot(range(1, len(loss) + 1), loss)
    plt.show()

class MyDataSet(Dataset):
    def __init__(self, x_train, y_train, normalize=True):
        #self.x_train = torch.from_numpy(x_train).requires_grad_(True)
        self.x_train = torch.from_numpy(x_train)
        self.y_train = torch.from_numpy(y_train)
        self.normalize = normalize

    def __getitem__(self, item):
        ts = self.x_train[item]
        label = int(self.y_train[item])

        if self.normalize:
            mean = torch.mean(ts)
            std = torch.std(ts)
            [(x - mean) / std for x in ts]
        return ts, label

    def __len__(self):
        return self.x_train.shape[0]

# nn.CrossEntropyLoss()
def loss_function(output, label):
    num_class = output.size(-1)
    one_hot = torch.zeros(BATCH_SIZE, num_class, 1)

    for b, l in enumerate(label):
        one_hot[b][l] = 1
    one_hot_inv = torch.ones(BATCH_SIZE, num_class, 1) - one_hot

    loss = torch.bmm(torch.log(output + 1e-10), one_hot) + torch.bmm(torch.log(torch.ones(num_class) - output + 1e-10), one_hot_inv)
    #print(torch.mean(loss))
    return torch.mean(-loss)


def train():
    # 后续从x_train抽出30%作为验证集
    x_train, y_train, x_test, y_test = read_data()
    category2real, real2category = redefine_category(y_train)
    y_train_transformed = transform_category(y_train, real2category)
    y_test_transformed = transform_category(y_test, real2category)

    input_size = len(x_train[0])
    class_num = len(category2real)
    # print(input_size)
    # print(class_num)

    train_set = MyDataSet(x_train, y_train_transformed, normalize=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    #dataiter = iter(train_loader)
    test_set = MyDataSet(x_test, y_test_transformed, normalize=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # for i, batch in enumerate(train_loader):
    #     print(i)
    #     # batch[0]为数据,batch[1]为标签
    #     print(batch[0], batch[1])
    #     break

    net = mWDN_RCF(input_size, class_num)
    #print(net)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    if torch.cuda.is_available():
        net = net.cuda()

    train_losses = []
    for e in tqdm(range(EPOCH)):
        net.train()
        print('\nEPOCH {e} of {E}'.format(e=e+1, E=EPOCH))
        train_loss = 0
        for i, batch in enumerate(train_loader):
            #print(i)
            input_data = batch[0].view(BATCH_SIZE, 1, -1)
            #print(input_data)
            c1, c2, c3 = net(input_data)
            #print(c3)
            #print(batch[1])
            loss = (1 * loss_function(c1, batch[1]) + 2 * loss_function(c2, batch[1]) + 3 * loss_function(c3, batch[1])) / 3
            # print(loss)
            W_mWDN1_H = net.mWDN1_H.weight.data
            W_mWDN1_L = net.mWDN1_L.weight.data
            W_mWDN2_H = net.mWDN2_H.weight.data
            W_mWDN2_L = net.mWDN2_L.weight.data
            W_mWDN3_H = net.mWDN3_H.weight.data
            W_mWDN3_L = net.mWDN3_L.weight.data

            L_loss = torch.norm((W_mWDN1_L - net.cmp_mWDN1_L), 2) + torch.norm((W_mWDN2_L - net.cmp_mWDN2_L), 2) + torch.norm((W_mWDN3_L - net.cmp_mWDN3_L), 2)
            H_loss = torch.norm((W_mWDN1_H - net.cmp_mWDN1_H), 2) + torch.norm((W_mWDN2_H - net.cmp_mWDN2_H), 2) + torch.norm((W_mWDN3_H - net.cmp_mWDN3_H), 2)

            # print('initial loss = {}'.format(loss.item()))
            # print('add loss = {}'.format(alpha * L_loss + beta * H_loss))
            loss += alpha * L_loss + beta * H_loss
            print('final loss = {}'.format(loss.item()))
            if str(loss.item()) == 'nan':
                break

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)

        net.eval()
        hit = 0
        for i, (data, label) in enumerate(train_loader):
            data = data.view(BATCH_SIZE, 1, -1)
            c1, c2, c3 = net(data)
            prediction = np.argmax(c3.detach().numpy(), axis=2).reshape(BATCH_SIZE)

            for k in range(BATCH_SIZE):
                if prediction[k] == label[k]:
                    hit += 1
        print('\nhit = {} of {}'.format(hit, BATCH_SIZE * len(train_loader)))

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

    final_loss = train_losses[-1]
    print(train_losses)

if __name__ == '__main__':
    train()

