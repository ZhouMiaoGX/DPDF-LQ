import torch.nn as nn
from .main_layer import CrossTransformer,Transformer,ConvModulOperationSpatialAttention
from .bert import BertTextEncoder
from einops import repeat

class LocalFusionClassifier(nn.Module):
    def __init__(self,args, dim=128, num_classes=1, fusion_depth=2, heads=8, mlp_dim=128):
        super(LocalFusionClassifier, self).__init__()
        args = args.model
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

        self.fusion_transformer = CrossTransformer(
            source_num_frames=58,  
            tgt_num_frames=58,
            dim=dim,
            depth=fusion_depth,
            heads=heads,
            mlp_dim=mlp_dim
        )


        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, num_classes)
        )

    def forward(self, video_feat, audio_feat,text_feat):

        text_feat = self.bertmodel(text_feat)

        video_feat = video_feat.unsqueeze(-1)
        video_feat=video_feat.permute(0, 2, 1, 3)
        video_feat=self.conv_att_v(video_feat)
        video_feat =video_feat.squeeze(-1)
        video_feat= video_feat.permute(0, 2, 1)

        audio_feat = self.proj_a(audio_feat)  # (batch_size, seq_len, dim)
        video_feat = self.proj_v(video_feat)  # (batch_size, seq_len, dim)
        text_feat = self.proj_l(text_feat)     # (batch_size, seq_len, dim)

        fused_feat = self.fusion_transformer(text_feat, audio_feat, video_feat)  # (batch_size, seq_len, dim)

        local_feat = fused_feat[:, 0, :]  # (batch_size, dim)

        return local_feat
def build_model():
    model =LocalFusionClassifier()
    return model