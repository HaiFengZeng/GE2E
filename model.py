import torch
from torch import nn
from torch.nn import functional as F
from ge2e_hparams import hparams
from torch.nn import init


class SpeakerEncoder(nn.Module):
    def __init__(self, input_size, n=hparams.N, m=hparams.M, hidden_size=768, project_size=256):
        super(SpeakerEncoder, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0))
        self.b = nn.Parameter(torch.tensor(-5.0))
        self.N = n
        self.M = m
        if hparams.mode == 'TD-SV':
            hidden_size = hparams.hidden_size_tdsv
            project_size = hparams.project_size_tdsv
        else:
            hidden_size = hparams.hidden_size_tisv
            project_size = hparams.project_size_tisv
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=0.5,
                             batch_first=False)
        self.project1 = nn.Linear(hidden_size, project_size)
        self.lstm2 = nn.LSTM(input_size=project_size, hidden_size=hidden_size, dropout=0.5,
                             batch_first=False)
        self.project2 = nn.Linear(hidden_size, project_size)
        self.lstm3 = nn.LSTM(input_size=project_size, hidden_size=hidden_size, dropout=0.5,
                             batch_first=False)
        self.project3 = nn.Linear(hidden_size, project_size)
        self.init()

    def init_lstm(self, lstm):
        for layer in lstm.all_weights:
            for p in layer:
                if len(p.size()) >= 2:
                    init.orthogonal_(p)

    def init(self):
        self.init_lstm(self.lstm1)
        self.init_lstm(self.lstm2)
        self.init_lstm(self.lstm3)
        init.normal_(self.project1.weight.data, 0, 0.02)
        init.normal_(self.project2.weight.data, 0, 0.02)
        init.normal_(self.project3.weight.data, 0, 0.02)

    def similarity_matrix(self, x):
        N, M = self.N, self.M
        # x [N*M,d] B=N*M,d is a vector
        yy = x.unsqueeze(0).repeat(N, 1, 1)
        c = torch.stack(x.split([M] * N), 0).mean(1, keepdim=True)
        cc = c.repeat(1, M * N, 1)
        cc = cc.permute(1, 0, 2)
        yy = yy.permute(1, 0, 2)
        sim = F.cosine_similarity(cc, yy, dim=-1)
        similarity = self.w * sim + self.b
        return similarity

    def forward(self, x, return_sim=True):

        x, (h1, c1) = self.lstm1(x)
        x = x.permute(1, 0, 2)
        x = self.project1(x)
        x = x.permute(1, 0, 2)
        x, (h2, c2) = self.lstm2(x)
        x = x.permute(1, 0, 2)
        x = self.project2(x)
        x = x.permute(1, 0, 2)
        x, (h3, c3) = self.lstm3(x)
        x = x.permute(1, 0, 2)
        x = self.project3(x)
        x = x.permute(1, 0, 2)
        x = x[-1, :, :]
        # l2 norm
        x = x / torch.norm(x)
        if not return_sim:
            return x, None
        sim = self.similarity_matrix(x)
        return x, sim


class GE2ELoss(nn.Module):
    def __init__(self, N=hparams.N, M=hparams.M, loss_type='softmax'):
        super(GE2ELoss, self).__init__()
        self.N = N
        self.M = M
        assert loss_type in ['softmax', 'contrast']
        self.loss_type = loss_type

    def softmax(self, x):
        N, M = self.N, self.M
        _x = torch.cat([x[i * M:(i + 1) * M, i:(i + 1)] for i in range(N)], 0)
        log_rs = torch.log(torch.sum(torch.exp(x), 1, keepdim=True) + 1e-6)
        return -torch.sum(_x - log_rs)

    def contrast(self, x):
        N, M = self.N, self.M
        c = x.split([M] * N, 0)
        c = torch.stack(c, 0)  # centroids [N,M,N]
        y = F.sigmoid(x) - F.sigmoid(x.max(-1)[0].unsqueeze(2).repeat(1, 1, N))
        return torch.sum(torch.sum(y, 1) * torch.eye(N))

    def forward(self, similarity_matrix):
        if self.loss_type == 'softmax':
            return self.softmax(similarity_matrix)
        else:
            return self.contrast(similarity_matrix)
