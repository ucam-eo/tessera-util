from .modules import TransformerEncoder, ProjectionHead
from .ssl_model import MultimodalBTModel

def build_ssl_model(config, device):
    # 注意：增强后 S2 数据通道数固定为 12（10 个原始波段 + 2 个 doy 特征），S1 数据通道数为 4（2+2）
    ###############哨兵2##################
    s2_backbone_ssl = TransformerEncoder(
        band_num=10,
        latent_dim=config['latent_dim'],
        # nhead=16,
        # num_encoder_layers=32,
        # dim_feedforward=512,        
        nhead=8,
        num_encoder_layers=2,
        dim_feedforward=4096,
        dropout=0.1,
        max_seq_len=20
    )
    ###############哨兵1##################
    s1_backbone_ssl = TransformerEncoder(
        band_num=2,
        latent_dim=config['latent_dim'],
        # nhead=16,
        # num_encoder_layers=32,
        # dim_feedforward=512,
        nhead=8,
        num_encoder_layers=2,
        dim_feedforward=4096,
        dropout=0.1,
        max_seq_len=20
    )   
    if config["fusion_method"] == "concat":
        input_dim_for_projector = config['latent_dim']
    else:
        input_dim_for_projector = config['latent_dim']
    projector_ssl = ProjectionHead(input_dim_for_projector,
                                    config["projector_hidden_dim"],
                                    config["projector_out_dim"])
    # 构建 SSL 模型以便载入 checkpoint
    ssl_model = MultimodalBTModel(s2_backbone_ssl, s1_backbone_ssl, projector_ssl,
                                    fusion_method=config["fusion_method"], return_repr=True).to(device)
    return ssl_model