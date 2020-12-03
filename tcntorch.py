# coding: utf-8

import os
import csv
import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (10, 4)  # (w, h)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['svg.fonttype'] = 'none'  # 'svgfont'

class VibrationDataset(Dataset):
    def __init__(self, root, csv_files, inputs=2, outputs=3, samples=0):
        """
        :param csv_files: файлы с данными
        :param outputs: число выходных переменных
        :param inputs: число входных переменных
        :param samples: максимальное число отсчетов (0 - все)
        """
        self.inputs = inputs
        self.outputs = outputs
        df = []
        for csv_file in csv_files:
            with open(os.path.join(root,csv_file), 'r') as f:
                reader = csv.reader(f)
                # первая строка содержит названия столбцов
                self.header = next(reader, None)

                cols = inputs + outputs
                for row in reader:
                    if row:  # может быть пустая строка в конце
                        df.append([float(x) for x in row[:cols]])
                    if samples and len(df) >= samples:
                        break
        self.dataframe = np.array(df)

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        # извлечение одной переменной по имени
        if isinstance(idx, str) and idx in self.header:
            return self.dataframe[:, self.header.index(idx)]
        # извлечение всех значений в заданный момент времени
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {
            'input': self.dataframe[idx, :self.inputs],
            'output': self.dataframe[idx, self.inputs:]
        }


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    # блок из двух сверток
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.dropout1,
                                 self.conv2, self.chomp2, self.relu2,
                                 self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs,
                                    1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.net(x)
        if res.size()[2] != out.size()[2]:
            res = res[:, :, -out.size()[2]:]
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    # сеть с заданным числом блоков TemporalBlock
    def __init__(self, n_inputs, n_outputs, n_channels, kernel_size=2,
                 dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(n_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = n_inputs if i == 0 else n_channels[i - 1]
            out_channels = n_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(n_channels[-1], n_outputs, 1)

    def forward(self, x):
        y = self.network(x)
        return self.final_conv(y)


def validate(model, loader):
    """
    :param model: обученная сеть
    :param loader: DataLoader с набором данных для валидации
    :return: среднеквадратическая ошибка
    """

    model.eval()
    total_vloss = 0
    total_batches = 0
    with torch.no_grad(): # не вычислять градиенты
        for batch in loader:
            x = batch['input'].unsqueeze(2).float()
            y = batch['output'].float()
            output = model(x).squeeze()
            vloss = F.mse_loss(output, y)
            total_batches += 1
            total_vloss += vloss.item()
    return total_vloss / total_batches


def main():

    epochs = 50 # число эпох обучения
    bsize = 64 # размер минипакета
    model_inputs = 2 # число входов
    model_outputs = 3 # число выходов

    data_folder = 'F16GVT_Files/BenchmarkData'
    train_files = ['F16Data_FullMSine_Level1.csv',
         'F16Data_FullMSine_Level3.csv']

    validation_files = ['F16Data_FullMSine_Level1.csv']

    train_dataset = VibrationDataset(
        data_folder,
        train_files,
        model_inputs,
        model_outputs
    )

    validation_dataset = VibrationDataset(
        data_folder,
        validation_files,
        model_inputs,
        model_outputs
    )

    model = TemporalConvNet(
        n_inputs=model_inputs,
        n_outputs=model_outputs,
        n_channels=[bsize]*4,
        kernel_size=3,
        dropout=0.1
    )
    model.to(device='cpu').float()

    vcurve = []


    # обучение будет производиться по минимуму среднеквадратической ошибки
    criterion = torch.nn.MSELoss()
    # алгоритм стохастической градиентной оптимизации Adam
    optimizer = torch.optim.Adam(
        model.parameters(), betas=(0.9, 0.999), lr=1e-3, weight_decay=0)

    validation_loader = DataLoader(validation_dataset, batch_size=1000)
    validation_loss = validate(model, validation_loader)
    print('Start training, MSE={}'.format(validation_loss))
    vcurve = [validation_loss]
    show_epoch = 1

    for epoch in range(epochs):
        # вспомогательный класс для разбивки данных на минипакеты
        train_loader = DataLoader(train_dataset, batch_size=bsize)
        model.train()
        for i, minibatch in enumerate(train_loader):
            # алгоритм оптимизации накапливает значения градиентов для всех
            # образцов из одного минипакета
            optimizer.zero_grad()
            # прямой проход
            x = minibatch['input'].unsqueeze(2).float()
            output = model(x).squeeze()  # прямой проход
            y = minibatch['output'].float()
            loss = criterion(output, y) # вычисление ошибки
            loss.backward()  # обратный проход
            optimizer.step()  # коррекция весов

        if epoch % show_epoch == 0:
            validation_loader = DataLoader(validation_dataset, batch_size=1000)
            validation_loss = validate(model, validation_loader)
            print('Epoch {}, MSE={}'.format(epoch, validation_loss))
            vcurve.append(validation_loss)

    torch.save({
        'epoch': epochs,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'vloss': validation_loss,
        'vcurve': vcurve
    }, 'model.pt')


def check_model(filename):
    n_channels = 64 # размер минипакета
    n_inputs = 2 # число входов
    n_outputs = 3 # число выходов

    model = TemporalConvNet(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_channels=[n_channels]*4,
        kernel_size=3,
        dropout=0.1
    )
    model.to(device='cpu').float()

    try:
        ckpt = torch.load(filename, map_location=lambda storage, loc: storage)
    except NotADirectoryError:
        raise Exception("Could not find model: " + filename)



    fig, ax = plt.subplots(1,2)
    print(plt.rcParams.keys())
    #plt.rcParams['axes.titlelocation'] = 'bottom'  # y is in axes-relative
    # co-ordinates.
    #plt.rcParams['axes.titlepad'] = -14  # pad is in points...

    ax[0].plot(ckpt['vcurve'], 'k')
    ax[0].set_xlabel('номер эпохи')
    ax[0].set_ylabel('среднеквадратическая ошибка')
    ax[0].set_title('а).')

    #plt.savefig("/Users/arseniy/GUAP-k53/CNN_book/рисунок-5-4"
    #            "-а.svg")
    #plt.savefig("/Users/arseniy/GUAP-k53/CNN_book/рисунок-5-4"
    #        "-а.png")

    model.load_state_dict(ckpt["model"])
    model.eval()

    data_folder = 'F16GVT_Files/BenchmarkData'
    eval_dataset = VibrationDataset(
        data_folder,
        ['F16Data_FullMSine_Level2_Validation.csv'],
        n_inputs,
        n_outputs
    )
    eval_loader = DataLoader(eval_dataset, batch_size=500)
    with torch.no_grad():  # не вычислять градиенты
        for i,batch in enumerate(eval_loader):
            x = batch['input'].unsqueeze(2).float()
            y = batch['output'].float()
            output = model(x).squeeze()
            print(F.mse_loss(output, y))
            ch=1

            ys = y[:,ch]
            ym = output[:,ch]

            if i == 6:
                ym *= np.linalg.norm(ys) / np.linalg.norm(ym)

                ax[1].plot(ys, 'k:')
                ax[1].plot(ym, 'k')
                ax[1].legend(('система', 'модель'))
                ax[1].set_xlabel('Время, отсчетов')
                ax[1].set_ylabel('Ускорение, g')
                ax[1].set_title('б).')
                print(i)
                break

    #plt.show()
    plt.savefig("/Users/arseniy/GUAP-k53/CNN_book/рисунок-5-4.svg")
    plt.savefig("/Users/arseniy/GUAP-k53/CNN_book/рисунок-5-4.png")

if __name__ == "__main__":
    #main()
    check_model('model.pt')