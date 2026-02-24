import torch
from torch import nn
from .main_layer import Transformer, CrossTransformer, DynamicGlobalLearnableQueryAttentionEncoder,ConvModulOperationSpatialAttention
from .bert import BertTextEncoder
from einops import repeat


class GlobalFusionClassifier(nn.Module):
    def __init__(self, args):
        super(GlobalFusionClassifier, self).__init__()

        args = args.model

        self.LearnableQuery = nn.Parameter(torch.ones(1, args.token_len, args.token_dim))

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=args.bert_pretrained)

        self.proj_l = nn.Sequential(
            nn.Linear(args.l_proj_dim, args.proj_dst_dim),
            Transformer(num_frames=args.l_proj_length, save_hidden=False, token_len=args.token_length, dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads, mlp_dim=args.proj_mlp_dim)
        )
        self.proj_a = nn.Sequential(
            nn.Linear(args.a_proj_dim, args.proj_dst_dim),
            Transformer(num_frames=args.a_proj_length, save_hidden=False, token_len=args.token_length, dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads, mlp_dim=args.proj_mlp_dim)
        )
        self.proj_v = nn.Sequential(
            nn.Linear(args.v_proj_dim, args.proj_dst_dim),
            Transformer(num_frames=args.v_proj_length, save_hidden=False, token_len=args.token_length, dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads, mlp_dim=args.proj_mlp_dim)
        )

        self.conv_att_v = ConvModulOperationSpatialAttention(args.v_proj_dim, kernel_size=3)


        self.l_encoder = Transformer(num_frames=args.token_length, save_hidden=True, token_len=None, dim=args.proj_input_dim, depth=args.DGLQA_depth-1, heads=args.l_enc_heads, mlp_dim=args.l_enc_mlp_dim)

        self.DGLQA_layer = DynamicGlobalLearnableQueryAttentionEncoder(dim=args.token_dim, depth=args.DGLQA_depth, heads=args.DGLQA_heads, dim_head=args.DGLQA_dim_head, dropout=args.DGLQA_droup)

        self.fusion_layer = CrossTransformer(source_num_frames=args.token_len, tgt_num_frames=args.token_len, dim=args.proj_input_dim, depth=args.fusion_layer_depth, heads=args.fusion_heads, mlp_dim=args.fusion_mlp_dim)

        self.regression_layer = nn.Sequential(
            nn.Linear(args.token_dim, 1)
        )

        self.gate_fc = nn.Sequential(
            nn.Linear(args.token_dim*3, args.token_dim),
            nn.Sigmoid()
        )
        

        self.dropout = nn.Dropout(p=0.2)
        

        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(args.token_dim, args.token_dim, kernel_size=3, groups=args.token_dim),
            nn.GELU(),
            nn.BatchNorm1d(args.token_dim)
        )        

    def forward(self, x_visual, x_audio, x_text):
        b = x_visual.size(0)

        LearnableQuery = repeat(self.LearnableQuery, '1 n d -> b n d', b=b)

        x_text = self.bertmodel(x_text)

        h_v_p = self.proj_v(x_visual)

        h_v = h_v_p[:, :self.LearnableQuery.shape[1]]
        h_a_p = self.proj_a(x_audio)
        h_a = h_a_p[:, :self.LearnableQuery.shape[1]]
        h_l_p = self.proj_l(x_text)
        h_l =h_l_p[:, :self.LearnableQuery.shape[1]]

        h_t_list = self.l_encoder(h_l)

        LearnableQuery = self.DGLQA_layer(h_t_list, h_a, h_v, LearnableQuery)


        LearnableQuery = LearnableQuery.permute(0, 2, 1)  # (b, dim, token_len)
        LearnableQuery = self.depthwise_conv(LearnableQuery)
        LearnableQuery = LearnableQuery.permute(0, 2, 1)  # (b, token_len, dim)
        
        
        feat = self.fusion_layer(LearnableQuery, h_t_list[-1])[:, 0]
        
        expanded_feat = feat.unsqueeze(1).expand(-1, LearnableQuery.size(1), -1)  # (b, token_len, dim)

        target_seq_len = LearnableQuery.size(1) 
        h_t_last = h_t_list[-1][:, :target_seq_len, :]  
        
        gate_input = torch.cat([LearnableQuery, h_t_last, expanded_feat], dim=-1)
        fusion_gate = self.gate_fc(gate_input)
        LearnableQuery = fusion_gate * LearnableQuery + (1 - fusion_gate) * h_t_last
        
        global_feat = self.fusion_layer(LearnableQuery, h_t_last)[:, 0]
        global_feat = self.dropout(global_feat)

        return global_feat


def build_model(args):
    model =GlobalFusionClassifier(args)

    return model



