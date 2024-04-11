import torch


class VisualEncoder(torch.nn.Module):

    def __init__(self, visual_feature_dim, num_heads=8, hidden_size=256):
        super(VisualEncoder, self).__init__()

        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=visual_feature_dim, nhead=num_heads,
                                                                  dim_feedforward=hidden_size)

        self.transformer_encoder = torch.nn.TransformerEncoder(self.transformer_layer, num_layers=1)

    def forward(self, x):
        transformer_output = self.transformer_encoder(x)

        transformer_output = transformer_output[:, -1, :]

        return transformer_output


if __name__ == '__main__':
    transformer_model = VisualEncoder(input_size=256)

    video_features = torch.rand((32, 10, 256))

    output_transformer = transformer_model(video_features, mode='target')

