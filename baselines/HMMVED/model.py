import torch
import torch.nn as nn
from visual_encoder import VisualEncoder


class HMMVED(nn.Module):

    def __init__(self, feature_size=768, latent_size=50):
        super(HMMVED, self).__init__()

        self.visual_dense_layer = VisualEncoder(visual_feature_dim=feature_size)

        self.visual_encoder = nn.Sequential(

            nn.Linear(feature_size, 512),

            nn.ReLU(),

            nn.Linear(512, 256),

            nn.ReLU()

        )

        self.textual_encoder = nn.Sequential(

            nn.Linear(feature_size, 512),

            nn.ReLU(),

            nn.Linear(512, 256),

            nn.ReLU()

        )

        self.textual_mu_layer = nn.Linear(256, latent_size)

        self.visual_mu_layer = nn.Linear(256, latent_size)

        self.textual_logvar_layer = nn.Linear(256, latent_size)

        self.visual_logvar_layer = nn.Linear(256, latent_size)

        self.decoder = nn.Sequential(

            nn.Linear(latent_size, 20),

            nn.ReLU(),

            nn.Linear(20, 1),

        )

    def generate_visual_feature_mu_std(self, x, logvar):
        mu = self.visual_mu_layer(x)

        std = torch.exp(0.5 * logvar)

        return mu, std

    def generate_textual_feature_mu_std(self, x, logvar):
        mu = self.textual_mu_layer(x)

        std = torch.exp(0.5 * logvar)

        return mu, std

    def generate_latent_representation_z(self, visual_mu, visual_std, textual_mu, textual_std):
        visual_prec = 1 / (visual_std ** 2)

        textual_prec = 1 / (textual_std ** 2)

        poe_mu = (visual_mu * visual_prec + textual_mu * textual_prec) / (visual_prec + textual_prec)

        poe_std = torch.sqrt(1 / (visual_prec + textual_prec))

        eps = torch.randn_like(poe_std)

        z = poe_mu + eps * poe_std

        return z

    def forward(self, visual_feature, textual_feature):
        visual_feature = self.visual_dense_layer(visual_feature)

        visual_feature = self.visual_encoder(visual_feature)

        textual_feature = self.textual_encoder(textual_feature)

        visual_logvar = self.visual_logvar_layer(visual_feature)

        textual_logvar = self.textual_logvar_layer(textual_feature)

        visual_mu, visual_std = self.generate_visual_feature_mu_std(visual_feature, visual_logvar)

        textual_mu, textual_std = self.generate_textual_feature_mu_std(textual_feature, textual_logvar)

        z = self.generate_latent_representation_z(visual_mu, visual_std, textual_mu, textual_std)

        output = self.decoder(z)

        return output




