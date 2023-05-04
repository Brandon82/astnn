import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class Code2VecEncoder(nn.Module):
    def __init__(self, input_dim, encode_dim, use_gpu, pretrained_weight=None):
        super(Code2VecEncoder, self).__init__()
        self.encode_dim = encode_dim
        self.fc = nn.Linear(input_dim, encode_dim)
        self.activation = F.relu
        self.use_gpu = use_gpu

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    def forward(self, x):
        encoded = self.activation(self.fc(x))
        return encoded


class BatchProgramClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, encode_dim, label_size, batch_size, use_gpu=True, pretrained_weight=None):
        super(BatchProgramClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.encode_dim = encode_dim
        self.label_size = label_size
        # class "Code2VecEncoder"
        self.encoder = Code2VecEncoder(input_dim, self.encode_dim, self.gpu, pretrained_weight)
        self.root2label = nn.Linear(self.encode_dim, self.label_size)
        # gru
        self.bigru = nn.GRU(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.label_size)
        # hidden
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.2)

    def init_hidden(self):
        if self.gpu is True:
            if isinstance(self.bigru, nn.LSTM):
                h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                return h0, c0
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

    def forward(self, x):
        lens = [len(item) for item in x]
        max_len = max(lens)

        encodes = []
        for i in range(self.batch_size):
            encodes.append(self.encoder(x[i]))

        encodes = torch.stack(encodes)
        encodes = encodes.view(self.batch_size, max_len, -1)

        # gru
        gru_out, hidden = self.bigru(encodes, self.hidden)

        gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)

        # linear
        y = self.hidden2label(gru_out)
        return y
