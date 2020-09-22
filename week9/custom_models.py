# models.py 파일과 논문을 바탕으로 빈칸을 채워주세요.
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.cnn_module = nn.Sequential(
            nn.Conv2d(1,64,3,1,1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            )
            
        self.rnn_model = nn.Sequential(
            nn.LSTM(512, 256, bidirectional=True),
            nn.Linear(256 * 2, 256),
            nn.LSTM(256, 256, bidirectional=True)
        )


        self.embedding = nn.Linear(256 * 2, 37)

    def forward(self, input):
        conv = self.cnn_module(input)
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        output, _ = self.rnn_model(conv)
        seq_len, batch, h_2 =  output.size()
        output = output.view(seq_len * batch, h_2)
        output = self.embedding(output)
        output = output.view(seq_len, batch, -1)
        return output

