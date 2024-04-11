import torch
import torch.nn as nn


class ContextualLSTM(nn.Module):
    def __init__(self, input_size_text, input_size_visual, hidden_size, output_size):
        super(ContextualLSTM, self).__init__()

        self.lstm_text = nn.LSTM(input_size=input_size_text, hidden_size=hidden_size, batch_first=True)

        self.lstm_visual = nn.LSTM(input_size=input_size_visual, hidden_size=hidden_size, batch_first=True)

        self.fc1 = nn.Linear(hidden_size * 2, 200)

        self.fc2 = nn.Linear(200, 1)

    def forward(self, x_visual, x_text):
        x_text = x_text.unsqueeze(1)

        _, (h_text, _) = self.lstm_text(x_text)

        _, (h_visual, _) = self.lstm_visual(x_visual)

        combined_features = torch.cat((h_text[-1], h_visual[-1]), dim=1)

        output = self.fc1(combined_features)

        output = self.fc2(output)

        return output

