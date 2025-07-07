# Contains CNN_BiLSTM_CTC class
import torch
import torch.nn as nn

class CNN_BiLSTM_CTC(nn.Module):
    def __init__(self, num_classes):
        super(CNN_BiLSTM_CTC, self).__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, 32, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [B, 64, 16, 64]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [B, 128, 16, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [B, 128, 8, 32]

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # [B, 256, 8, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [B, 256, 4, 16]
        )

        # BiLSTM layers
        self.rnn = nn.LSTM(
            input_size=256 * 4,  # because H=4, W=16 → features reshaped to (B, W, C*H)
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Final linear layer for classification
        self.fc = nn.Linear(512, num_classes)  # 256*2 for BiLSTM output

    def forward(self, x):
        x = self.cnn(x)  # [B, C, H, W] → [B, 256, 4, 16]
        b, c, h, w = x.size()

        x = x.permute(0, 3, 1, 2)  # [B, W, C, H]
        x = x.reshape(b, w, c * h)  # [B, W, C*H]

        rnn_out, _ = self.rnn(x)  # [B, W, 512]
        output = self.fc(rnn_out)  # [B, W, num_classes]

        output = output.permute(1, 0, 2)  # [T, B, C] for CTC Loss
        return output
