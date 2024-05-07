import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, alpha, frame_num, feature_dim=768):

        super(Model, self).__init__()

        self.alpha = alpha

        self.frame_num = frame_num

        self.feature_dim = feature_dim

        self.visual_embedding = nn.Linear(feature_dim, feature_dim)

        self.textual_embedding = nn.Linear(feature_dim, feature_dim)

        self.retrieval_visual_embedding = nn.Linear(feature_dim, feature_dim)

        self.retrieval_textual_embedding = nn.Linear(feature_dim, feature_dim)

        self.tanh = nn.Tanh()

        self.dual_attention_linear_1 = nn.Linear(feature_dim * 2, feature_dim)

        self.dual_attention_linear_2 = nn.Linear(feature_dim * 2, feature_dim)

        self.retrieval_dual_attention_linear_1 = nn.Linear(feature_dim * 2, feature_dim)

        self.retrieval_dual_attention_linear_2 = nn.Linear(feature_dim * 2, feature_dim)

        self.cross_modal_linear_1 = nn.Linear(feature_dim * 2, feature_dim)

        self.cross_modal_linear_2 = nn.Linear(feature_dim * 2, feature_dim)

        self.retrieval_cross_modal_linear_1 = nn.Linear(feature_dim * 2, feature_dim)

        self.retrieval_cross_modal_linear_2 = nn.Linear(feature_dim * 2, feature_dim)

        self.uni_modal_linear_1 = nn.Linear(feature_dim, 1)

        self.uni_modal_linear_2 = nn.Linear(feature_dim, 1)

        self.retrieval_uni_modal_linear_1 = nn.Linear(feature_dim, 1)

        self.retrieval_uni_modal_linear_2 = nn.Linear(feature_dim, 1)

        self.prediction_module = nn.Sequential(
            nn.Linear(feature_dim * 10, 800),
            nn.ReLU(),
            nn.Linear(800, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

        self.label_embedding_linear = nn.Linear(1, feature_dim)

    def cross_modal_attention(self, visual_feature, textual_feature, visual_feature_emb, textual_feature_emb):

        S = (torch.matmul(visual_feature_emb, textual_feature_emb.transpose(1, 2)) / self.feature_dim)

        T_p = torch.matmul(torch.softmax(S, dim=1), textual_feature)

        V_p = torch.matmul(torch.softmax(S.transpose(1, 2), dim=1), visual_feature)

        T_n = torch.matmul((-self.alpha) * torch.softmax(S, dim=1), textual_feature)

        V_n = torch.matmul((-self.alpha) * torch.softmax(S.transpose(1, 2), dim=1), visual_feature)

        T_star = self.dual_attention_linear_1(torch.cat([T_p, T_n], dim=2))

        V_star = self.dual_attention_linear_2(torch.cat([V_p, V_n], dim=2))

        T_star = self.tanh(T_star)

        V_star = self.tanh(V_star)

        V_f = self.cross_modal_linear_1(torch.cat([visual_feature, T_star], dim=2))

        T_f = self.cross_modal_linear_2(torch.cat([textual_feature, V_star], dim=2))

        V_f = self.tanh(V_f)

        T_f = self.tanh(T_f)

        return V_f, T_f

    def retrieval_cross_modal_attention(self, visual_feature, textual_feature, visual_feature_emb, textual_feature_emb):

        S = (torch.matmul(visual_feature_emb, textual_feature_emb.transpose(1, 2)) / self.feature_dim)

        T_p = torch.matmul(torch.softmax(S, dim=1), textual_feature)

        V_p = torch.matmul(torch.softmax(S.transpose(1, 2), dim=1), visual_feature)

        T_n = torch.matmul((-self.alpha) * torch.softmax(S, dim=1), textual_feature)

        V_n = torch.matmul((-self.alpha) * torch.softmax(S.transpose(1, 2), dim=1), visual_feature)

        T_star = self.retrieval_dual_attention_linear_1(torch.cat([T_p, T_n], dim=2))

        V_star = self.retrieval_dual_attention_linear_2(torch.cat([V_p, V_n], dim=2))

        T_star = self.tanh(T_star)

        V_star = self.tanh(V_star)

        V_f = self.retrieval_cross_modal_linear_1(torch.cat([visual_feature, T_star], dim=2))

        T_f = self.retrieval_cross_modal_linear_2(torch.cat([textual_feature, V_star], dim=2))

        V_f = self.tanh(V_f)

        T_f = self.tanh(T_f)

        return V_f, T_f

    def uni_modal_attention(self, V_f, T_f):

        alpha_v = torch.softmax(self.uni_modal_linear_1(V_f) / self.feature_dim, dim=1)

        V_f_star = torch.matmul(alpha_v.transpose(1, 2), V_f)

        alpha_t = torch.softmax(self.uni_modal_linear_2(T_f) / self.feature_dim, dim=1)

        T_f_star = torch.matmul(alpha_t.transpose(1, 2), T_f)

        return V_f_star, T_f_star

    def retrieval_uni_modal_attention(self, V_f, T_f):

        alpha_v = torch.softmax(self.retrieval_uni_modal_linear_1(V_f) / self.feature_dim, dim=1)

        V_f_star = torch.matmul(alpha_v.transpose(1, 2), V_f)

        alpha_t = torch.softmax(self.retrieval_uni_modal_linear_2(T_f) / self.feature_dim, dim=1)

        T_f_star = torch.matmul(alpha_t.transpose(1, 2), T_f)

        return V_f_star, T_f_star

    def retrieval_aggregation(self, similarity, retrieved_visual_feature, retrieved_textual_feature, retrieved_label):

        similarity = torch.softmax(similarity, dim=1)

        similarity = similarity.unsqueeze(2)

        retrieved_aggregated_label = torch.matmul(similarity.transpose(1, 2), retrieved_label)

        retrieved_textual_aggregated_feature = torch.matmul(similarity.transpose(1, 2), retrieved_textual_feature)

        for i in range(self.frame_num):

            retrieved_visual_feature_tmp = retrieved_visual_feature[:, :, i, :]

            retrieved_visual_feature_tmp = torch.matmul(similarity.transpose(1, 2), retrieved_visual_feature_tmp)

            if i == 0:
                retrieved_visual_aggregated_feature = retrieved_visual_feature_tmp

            else:
                retrieved_visual_aggregated_feature = torch.cat(
                    [retrieved_visual_aggregated_feature, retrieved_visual_feature_tmp], dim=1)

        return retrieved_visual_aggregated_feature, retrieved_textual_aggregated_feature, retrieved_aggregated_label

    def forward(self, visual_feature, textual_feature, similarity, retrieved_visual_feature, retrieved_textual_feature,
                retrieved_label):

        visual_feature_emb = self.visual_embedding(visual_feature)

        visual_feature_emb = self.tanh(visual_feature_emb)

        textual_feature_emb = self.textual_embedding(textual_feature)

        textual_feature_emb = self.tanh(textual_feature_emb)

        V_f, T_f = self.cross_modal_attention(visual_feature, textual_feature, visual_feature_emb, textual_feature_emb)

        V_f_star, T_f_star = self.uni_modal_attention(V_f, T_f)

        retrieved_visual_aggregated_feature, retrieved_textual_aggregated_feature, retrieved_aggregated_label = self.retrieval_aggregation(
            similarity, retrieved_visual_feature, retrieved_textual_feature, retrieved_label)

        retrieved_visual_feature_emb = self.retrieval_visual_embedding(retrieved_visual_aggregated_feature)

        retrieved_visual_feature_emb = self.tanh(retrieved_visual_feature_emb)

        retrieved_textual_feature_emb = self.retrieval_textual_embedding(retrieved_textual_aggregated_feature)

        retrieved_textual_feature_emb = self.tanh(retrieved_textual_feature_emb)

        V_f_, T_f_ = self.retrieval_cross_modal_attention(retrieved_visual_aggregated_feature,
                                                          retrieved_textual_aggregated_feature,
                                                          retrieved_visual_feature_emb, retrieved_textual_feature_emb)

        V_f_star_, T_f_star_ = self.retrieval_uni_modal_attention(V_f_, T_f_)

        retrieved_aggregated_label_embedding = self.label_embedding_linear(retrieved_aggregated_label)

        retrieved_aggregated_label_embedding = self.relu(retrieved_aggregated_label_embedding)

        r_v_v = torch.mul(V_f_star, V_f_star_)

        r_v_t = torch.mul(V_f_star, T_f_star_)

        r_v_l = torch.mul(V_f_star, retrieved_aggregated_label_embedding)

        r_t_l = torch.mul(T_f_star, retrieved_aggregated_label_embedding)

        r_t_v = torch.mul(T_f_star, V_f_star_)

        r_t_t = torch.mul(T_f_star, T_f_star_)

        output = self.prediction_module(
            torch.cat([T_f_star, V_f_star, T_f_star_, V_f_star_, r_v_v, r_v_t, r_t_v, r_t_t, r_v_l, r_t_l], dim=2))

        output = output.squeeze(2)

        return output
