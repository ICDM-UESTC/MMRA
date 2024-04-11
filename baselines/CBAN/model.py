import torch
import torch.nn as nn


class CBAN(nn.Module):

    def __init__(self, feature_dim=768):
        super(CBAN, self).__init__()

        self.feature_dim = feature_dim

        self.visual_embedding = nn.Linear(feature_dim, feature_dim)

        self.textual_embedding = nn.Linear(feature_dim, feature_dim)

        self.tahn = nn.Tanh()

        self.dual_attention_linear_1 = nn.Linear(feature_dim * 2, feature_dim)

        self.dual_attention_linear_2 = nn.Linear(feature_dim * 2, feature_dim)

        self.cross_modal_linear_1 = nn.Linear(feature_dim * 2, feature_dim)

        self.cross_modal_linear_2 = nn.Linear(feature_dim * 2, feature_dim)

        self.uni_modal_linear_1 = nn.Linear(feature_dim, 1)

        self.uni_modal_linear_2 = nn.Linear(feature_dim, 1)

        self.predict_linear = nn.Linear(feature_dim * 2, 1)

        self.add_attention_linear_1 = nn.Linear(feature_dim, 1)

        self.add_attention_linear_2 = nn.Linear(feature_dim, 1)

        self.add_attention_linear_3 = nn.Linear(feature_dim, 1)

        self.add_attention_matrix_1 = nn.Parameter(torch.randn(4, feature_dim))

        self.add_attention_matrix_2 = nn.Parameter(torch.randn(1, feature_dim))

    def forward(self, visual_feature, textual_feature):
        textual_feature = textual_feature.unsqueeze(1)

        visual_feature_emb = self.visual_embedding(visual_feature)

        visual_feature_emb = self.tahn(visual_feature_emb)

        textual_feature_emb = self.textual_embedding(textual_feature)

        textual_feature_emb = self.tahn(textual_feature_emb)

        S = self.tahn(self.add_attention_linear_1(visual_feature_emb + textual_feature_emb))

        T_p = torch.matmul(torch.softmax(S, dim=1), textual_feature)

        V_p = torch.matmul(torch.softmax(S.transpose(1, 2), dim=1), visual_feature)

        T_n = torch.matmul(-1 * torch.softmax(S, dim=1), textual_feature)

        V_n = torch.matmul(-1 * torch.softmax(S.transpose(1, 2), dim=1), visual_feature)

        T_star = self.dual_attention_linear_1(torch.cat([T_p, T_n], dim=2))

        V_star = self.dual_attention_linear_2(torch.cat([V_p, V_n], dim=2))

        T_star = self.tahn(T_star)

        V_star = self.tahn(V_star)

        V_f = self.cross_modal_linear_1(torch.cat([visual_feature, T_star], dim=2))

        T_f = self.cross_modal_linear_2(torch.cat([textual_feature, V_star], dim=2))

        V_f = self.tahn(V_f)

        T_f = self.tahn(T_f)

        alpha_v = self.tahn(self.add_attention_linear_2(V_f + self.add_attention_matrix_1))

        V_f_star = torch.matmul(alpha_v.transpose(1, 2), V_f)

        alpha_t = self.tahn(self.add_attention_linear_3(T_f + self.add_attention_matrix_2))

        T_f_star = torch.matmul(alpha_t.transpose(1, 2), T_f)

        output = self.predict_linear(torch.cat([V_f_star, T_f_star], dim=2))

        output = output.squeeze(2)

        return output

