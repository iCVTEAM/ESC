# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
import logging


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class Head_A(nn.Module):
    def __init__(self, in_channels, embedding_dim, num_classes):
        super(Head_A, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels


        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.dropout = nn.Dropout2d(0.1)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs # c1 4x64x64x64 c2 4x128x32x32 c3 4x320x16x16 c4 4x512x8x8

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]) # 4x256x8x8
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3]) # 4x256x16x16
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3]) # 4x256x32x32
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3]) # 4x256x64x64

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1)) # 4x256x64x64

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
    

class Head_B(nn.Module):
    def __init__(self, in_channels, embedding_dim, num_classes):
        super(Head_B, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels


        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.dropout = nn.Dropout2d(0.1)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )        

        self.linear_rf4 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        self.linear_rf3 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        self.linear_rf2 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        self.linear_rf1 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)

        self.rf_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_df4 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        self.linear_df3 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        self.linear_df2 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        self.linear_df1 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)

        self.df_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim * 2, self.num_classes, kernel_size=1)

        self.attn_1 = nn.MultiheadAttention(embedding_dim, num_heads=8, dropout=0.0, batch_first=True)
        self.attn_2 = nn.MultiheadAttention(embedding_dim, num_heads=8, dropout=0.0, batch_first=True)
        self.k_noise = nn.Embedding(2, embedding_dim)
        self.v_noise = nn.Embedding(2, embedding_dim)

        self.attn_0 = nn.MultiheadAttention(embedding_dim, num_heads=8, dropout=0.0, batch_first=True)
        self.noise_0 = nn.Embedding(2, embedding_dim)

    def query_edge(self, rgb_feature, embeddings):
        # import pdb; pdb.set_trace()
        B, H, W = rgb_feature.shape[0], rgb_feature.shape[2], rgb_feature.shape[3]
        feature = rgb_feature.permute(0, 2, 3, 1).reshape(B * H * W, 1, -1)
        embedding_1, embedding_2 = embeddings[0].permute(0, 2, 3, 1).reshape(B * H * W, 1, -1), embeddings[1].permute(0, 2, 3, 1).reshape(B * H * W, 1, -1)
        q = feature
        noise_k = self.noise_0.weight[0] + q
        noise_v = self.noise_0.weight[1] + q
        k = torch.cat([noise_k, embedding_1, embedding_2], dim=1)
        v = torch.cat([noise_v, embedding_1, embedding_2], dim=1)
        refined_feature = feature + self.attn_0(q, k, v)[0]
        return refined_feature.reshape(B, H, W, -1).permute(0, 3, 1, 2)

    def fuse_edge(self, features, confidences, similarity):
        # import pdb; pdb.set_trace()
        B, H, W = features[0].shape[0], features[0].shape[2], features[0].shape[3]
        feature_1, feature_2 = features[0].permute(0, 2, 3, 1).reshape(B * H * W, 1, -1), features[1].permute(0, 2, 3, 1).reshape(B * H * W, 1, -1)
        confidence_1, confidence_2, similarity = confidences[0].reshape(B * H * W, 1, 1), confidences[1].reshape(B * H * W, 1, 1), similarity.reshape(B * H * W, 1, 1)
        complement_1, complement_2 = 1 - confidence_1, 1 - confidence_2

        q = feature_1
        noise_k = self.k_noise.weight[0] + q
        noise_v = self.v_noise.weight[0] + q
        k = torch.cat([noise_k, torch.mul(q, complement_1)], dim=1)
        v = torch.cat([noise_v, feature_2], dim=1)
        refined_feature_1 = feature_1 + self.attn_1(q, k, v)[0]

        q = feature_2
        noise_k = self.k_noise.weight[1] + q
        noise_v = self.v_noise.weight[1] + q
        k = torch.cat([noise_k, torch.mul(q, complement_2)], dim=1)
        v = torch.cat([noise_v, feature_1], dim=1)
        refined_feature_2 = feature_2 + self.attn_2(q, k, v)[0]

        weight_1 = confidence_1 / (confidence_1 + confidence_2)
        weight_2 = confidence_2 / (confidence_1 + confidence_2)
        weighted_feature = torch.mul(refined_feature_1, weight_1) + torch.mul(refined_feature_2, weight_2)
        # weighted_feature = (refined_feature_1 + refined_feature_2) / 2
        return weighted_feature.reshape(B, H, W, -1).permute(0, 3, 1, 2)

    def forward(self, inputs, edge_encoded_features, embeddings, confidences, similarity):
        c1, c2, c3, c4 = inputs # c1 4x64x64x64 c2 4x128x32x32 c3 4x320x16x16 c4 4x512x8x8
        rf1, rf2, rf3, rf4 = edge_encoded_features[0]
        df1, df2, df3, df4 = edge_encoded_features[1]

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]) # 4x256x8x8
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3]) # 4x256x16x16
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3]) # 4x256x32x32
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3]) # 4x256x64x64

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1)) # 4x256x64x64

        ############## MLP decoder on RF1-RF4 ###########
        _rf4 = self.linear_rf4(rf4).permute(0,2,1).reshape(n, -1, rf4.shape[2], rf4.shape[3]) # 4x256x8x8
        _rf4 = F.interpolate(_rf4, size=rf1.size()[2:],mode='bilinear',align_corners=False)

        _rf3 = self.linear_rf3(rf3).permute(0,2,1).reshape(n, -1, rf3.shape[2], rf3.shape[3]) # 4x256x16x16
        _rf3 = F.interpolate(_rf3, size=rf1.size()[2:],mode='bilinear',align_corners=False)

        _rf2 = self.linear_rf2(rf2).permute(0,2,1).reshape(n, -1, rf2.shape[2], rf2.shape[3]) # 4x256x32x32
        _rf2 = F.interpolate(_rf2, size=rf1.size()[2:],mode='bilinear',align_corners=False)

        _rf1 = self.linear_rf1(rf1).permute(0,2,1).reshape(n, -1, rf1.shape[2], rf1.shape[3]) # 4x256x64x64

        _rf = self.rf_fuse(torch.cat([_rf4, _rf3, _rf2, _rf1], dim=1)) # 4x256x64x64

        ############## MLP decoder on DF1-DF4 ###########
        _df4 = self.linear_df4(df4).permute(0,2,1).reshape(n, -1, df4.shape[2], df4.shape[3]) # 4x256x8x8
        _df4 = F.interpolate(_df4, size=df1.size()[2:],mode='bilinear',align_corners=False)

        _df3 = self.linear_df3(df3).permute(0,2,1).reshape(n, -1, df3.shape[2], df3.shape[3]) # 4x256x16x16
        _df3 = F.interpolate(_df3, size=df1.size()[2:],mode='bilinear',align_corners=False)

        _df2 = self.linear_df2(df2).permute(0,2,1).reshape(n, -1, df2.shape[2], df2.shape[3]) # 4x256x32x32
        _df2 = F.interpolate(_df2, size=df1.size()[2:],mode='bilinear',align_corners=False)

        _df1 = self.linear_df1(df1).permute(0,2,1).reshape(n, -1, df1.shape[2], df1.shape[3]) # 4x256x64x64

        _df = self.df_fuse(torch.cat([_df4, _df3, _df2, _df1], dim=1)) # 4x256x64x64

        z = self.fuse_edge((_rf, _df), confidences, similarity)

        y = self.query_edge(_c, embeddings)

        x = torch.cat([y, z], dim=1)

        x = self.dropout(x)
        x = self.linear_pred(x)

        return x