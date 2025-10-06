import torch
import torch.nn as nn
import torch.nn.functional as F
from ldm.util import count_params
import math

class SimpleProjector(nn.Module):
    """簡單的全局投影器 - 保留原始設計作為baseline"""
    def __init__(self, input_dim=2048, output_dim=1024, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.projection(x)

class SequenceProjector(nn.Module):
    """將全局嵌入擴展為序列，模擬CLAP的token結構"""
    def __init__(self, input_dim=2048, output_dim=1024, seq_len=77, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        
        # 投影到高維空間
        self.expand_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim * seq_len)
        )
        
        # 位置編碼
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, output_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 層標準化
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        B = x.shape[0]
        # 擴展到序列 [B, 2048] -> [B, 77*1024]
        expanded = self.expand_proj(x)
        # Reshape [B, 77*1024] -> [B, 77, 1024]
        sequence = expanded.view(B, self.seq_len, self.output_dim)
        # 添加位置編碼
        sequence = sequence + self.pos_embed
        # 標準化
        sequence = self.norm(sequence)
        return sequence

class TransformerProjector(nn.Module):
    """使用Transformer生成序列表示"""
    def __init__(self, input_dim=2048, output_dim=1024, seq_len=77, 
                 num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        
        # 輸入投影
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # 可學習的查詢tokens
        self.query_tokens = nn.Parameter(torch.zeros(1, seq_len, output_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        
        # Transformer解碼器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # 最終層標準化
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        B = x.shape[0]
        # 投影輸入 [B, 2048] -> [B, 1, 1024]
        memory = self.input_proj(x).unsqueeze(1)
        
        # 擴展查詢tokens
        queries = self.query_tokens.expand(B, -1, -1)
        
        # Transformer處理
        # 需要轉置以符合PyTorch的預期格式 (seq_len, batch, dim)
        queries = queries.transpose(0, 1)
        memory = memory.transpose(0, 1)
        
        output = self.transformer(queries, memory)
        
        # 轉回 (batch, seq_len, dim)
        output = output.transpose(0, 1)
        
        # 標準化
        output = self.norm(output)
        
        return output

class HybridProjector(nn.Module):
    """混合方法：結合全局和局部信息"""
    def __init__(self, input_dim=2048, output_dim=1024, seq_len=77, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        
        # 全局特徵投影
        self.global_proj = nn.Linear(input_dim, output_dim)
        
        # 生成序列的基礎模板
        self.sequence_generator = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, (seq_len - 1) * output_dim)  # 留一個位置給全局特徵
        )
        
        # 調制網絡
        self.modulation = nn.Sequential(
            nn.Linear(output_dim, seq_len),
            nn.Sigmoid()
        )
        
        # 位置編碼
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, output_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B = x.shape[0]
        
        # 全局特徵
        global_feat = self.global_proj(x)  # [B, 1024]
        
        # 生成序列
        sequence = self.sequence_generator(x)  # [B, 76*1024]
        sequence = sequence.view(B, self.seq_len - 1, self.output_dim)  # [B, 76, 1024]
        
        # 組合：第一個位置是全局特徵（類似CLS token）
        full_sequence = torch.cat([
            global_feat.unsqueeze(1),  # [B, 1, 1024]
            sequence  # [B, 76, 1024]
        ], dim=1)  # [B, 77, 1024]
        
        # 調制權重
        mod_weights = self.modulation(global_feat)  # [B, 77]
        mod_weights = mod_weights.unsqueeze(-1)  # [B, 77, 1]
        
        # 應用調制和位置編碼
        full_sequence = full_sequence * mod_weights + self.pos_embed
        
        # 標準化和dropout
        full_sequence = self.norm(full_sequence)
        full_sequence = self.dropout(full_sequence)
        
        return full_sequence

class AttentionProjector(nn.Module):
    """使用交叉注意力融合多模態信息"""
    def __init__(self, input_dim=2048, output_dim=1024, seq_len=77, 
                 num_heads=8, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        
        # 假設輸入是圖像和文本嵌入的串聯
        self.image_proj = nn.Linear(1024, output_dim)
        self.text_proj = nn.Linear(1024, output_dim)
        
        # 序列模板
        self.sequence_template = nn.Parameter(torch.zeros(1, seq_len, output_dim))
        nn.init.trunc_normal_(self.sequence_template, std=0.02)
        
        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim)
        )
        
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B = x.shape[0]
        
        # 分離圖像和文本嵌入（假設前1024是圖像，後1024是文本）
        image_emb = x[:, :1024]
        text_emb = x[:, 1024:]
        
        # 投影
        image_feat = self.image_proj(image_emb).unsqueeze(1)  # [B, 1, 1024]
        text_feat = self.text_proj(text_emb).unsqueeze(1)    # [B, 1, 1024]
        
        # 組合鍵值對
        kv = torch.cat([image_feat, text_feat], dim=1)  # [B, 2, 1024]
        
        # 擴展序列模板作為查詢
        queries = self.sequence_template.expand(B, -1, -1)  # [B, 77, 1024]
        
        # 交叉注意力
        attn_out, _ = self.cross_attn(queries, kv, kv)
        queries = queries + self.dropout(attn_out)
        queries = self.norm1(queries)
        
        # FFN
        ffn_out = self.ffn(queries)
        output = queries + self.dropout(ffn_out)
        output = self.norm2(output)
        
        return output

class ConvProjector(nn.Module):
    """使用1D卷積生成序列"""
    def __init__(self, input_dim=2048, output_dim=1024, seq_len=77, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        
        # 初始投影
        self.input_proj = nn.Linear(input_dim, output_dim * 8)
        
        # 1D卷積層生成序列
        self.conv_layers = nn.Sequential(
            # [B, 8*1024, 1] -> [B, 4*1024, 19]
            nn.ConvTranspose1d(output_dim * 8, output_dim * 4, 
                              kernel_size=19, stride=1, padding=0),
            nn.GELU(),
            nn.Dropout(dropout),
            # [B, 4*1024, 19] -> [B, 2*1024, 39]
            nn.ConvTranspose1d(output_dim * 4, output_dim * 2,
                              kernel_size=21, stride=1, padding=0),
            nn.GELU(),
            nn.Dropout(dropout),
            # [B, 2*1024, 39] -> [B, 1024, 77]
            nn.ConvTranspose1d(output_dim * 2, output_dim,
                              kernel_size=39, stride=1, padding=0),
        )
        
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        B = x.shape[0]
        
        # 初始投影
        x = self.input_proj(x)  # [B, 8*1024]
        x = x.unsqueeze(-1)  # [B, 8*1024, 1]
        
        # 1D卷積生成序列
        x = self.conv_layers(x)  # [B, 1024, 77]
        
        # 轉置並標準化
        x = x.transpose(1, 2)  # [B, 77, 1024]
        x = self.norm(x)
        
        return x


class AudioAwareProjector(nn.Module):
    """
    音頻感知的投影器 - 使用 Conformer 風格的架構
    結合 CNN 的局部建模能力和 Transformer 的全局建模能力
    """
    def __init__(self, input_dim=2048, output_dim=1024, seq_len=77, 
                 num_layers=4, num_heads=8, dropout=0.1, 
                 conv_kernel_size=31, expansion_factor=4):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        
        # 1. 模態分離投影
        self.image_proj = nn.Sequential(
            nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        self.text_proj = nn.Sequential(
            nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        # 2. 音頻先驗編碼
        # 頻率感知編碼 (模擬音頻的頻率特性)
        self.freq_embed = nn.Parameter(torch.randn(1, seq_len, output_dim // 4))
        # 時序演化模式 (模擬音頻的時序動態)
        self.temporal_embed = nn.Parameter(torch.randn(1, seq_len, output_dim // 4))
        # 節奏模式 (模擬音頻的節奏特性)
        self.rhythm_embed = nn.Parameter(torch.randn(1, seq_len // 4, output_dim // 4))
        
        # 3. 初始序列生成器
        self.init_sequence = nn.Parameter(torch.randn(1, seq_len, output_dim // 4))
        
        # 4. Conformer 風格的編碼器層
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(
                dim=output_dim,
                num_heads=num_heads,
                conv_kernel_size=conv_kernel_size,
                expansion_factor=expansion_factor,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # 5. 模態融合層
        self.modal_fusion = ModalFusionLayer(
            dim=output_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 6. 最終投影
        self.final_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 初始化
        nn.init.trunc_normal_(self.freq_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        nn.init.trunc_normal_(self.rhythm_embed, std=0.02)
        nn.init.trunc_normal_(self.init_sequence, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # 1. 分離並投影模態
        image_emb = x[:, :1024]
        text_emb = x[:, 1024:]
        
        image_feat = self.image_proj(image_emb)  # [B, 1024]
        text_feat = self.text_proj(text_emb)     # [B, 1024]
        
        # 2. 構建初始序列
        # 組合音頻先驗
        freq_pattern = self.freq_embed.expand(B, -1, -1)
        temporal_pattern = self.temporal_embed.expand(B, -1, -1)
        
        # 擴展節奏模式到完整序列長度
        rhythm_pattern = F.interpolate(
            self.rhythm_embed.transpose(1, 2),
            size=self.seq_len,
            mode='linear',
            align_corners=True
        ).transpose(1, 2).expand(B, -1, -1)
        
        # 組合所有先驗
        audio_priors = torch.cat([
            freq_pattern,
            temporal_pattern,
            rhythm_pattern,
            self.init_sequence.expand(B, -1, -1)
        ], dim=-1)  # [B, 77, 1024]
        
        # 3. 注入全局信息
        # 使用廣播機制將全局特徵加到每個位置
        global_info = (image_feat + text_feat).unsqueeze(1) * 0.5  # [B, 1, 1024]
        sequence = audio_priors + global_info  # [B, 77, 1024]
        
        # 4. 通過 Conformer 層處理
        for layer in self.conformer_layers:
            sequence = layer(sequence)
        
        # 5. 模態融合
        sequence = self.modal_fusion(sequence, image_feat, text_feat)
        
        # 6. 最終投影
        output = self.final_proj(sequence)
        
        return output


class ConformerBlock(nn.Module):
    """
    Conformer 塊：結合自注意力和卷積
    架構：FFN → Self-Attention → Conv → FFN
    """
    def __init__(self, dim, num_heads=8, conv_kernel_size=31, 
                 expansion_factor=4, dropout=0.1):
        super().__init__()
        
        # 第一個 Feed Forward
        self.ffn1 = FeedForward(dim, expansion_factor, dropout)
        
        # 多頭自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.attn_dropout = nn.Dropout(dropout)
        
        # 深度卷積模塊
        self.conv_module = ConvModule(dim, conv_kernel_size, dropout)
        
        # 第二個 Feed Forward
        self.ffn2 = FeedForward(dim, expansion_factor, dropout)
        
        # 層縮放參數
        self.layer_scale_1 = nn.Parameter(torch.ones(dim) * 0.1)
        self.layer_scale_2 = nn.Parameter(torch.ones(dim) * 0.1)
        self.layer_scale_3 = nn.Parameter(torch.ones(dim) * 0.1)
        self.layer_scale_4 = nn.Parameter(torch.ones(dim) * 0.1)
    
    def forward(self, x):
        # FFN1
        x = x + self.layer_scale_1 * self.ffn1(x)
        
        # Self-Attention
        attn_out = self.attn_norm(x)
        attn_out, _ = self.self_attn(attn_out, attn_out, attn_out)
        x = x + self.layer_scale_2 * self.attn_dropout(attn_out)
        
        # Conv Module
        x = x + self.layer_scale_3 * self.conv_module(x)
        
        # FFN2
        x = x + self.layer_scale_4 * self.ffn2(x)
        
        return x


class FeedForward(nn.Module):
    """前向傳播模塊"""
    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * expansion_factor)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * expansion_factor, dim)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class ConvModule(nn.Module):
    """卷積模塊：用於捕捉局部模式"""
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
        # Pointwise conv
        self.pointwise_conv1 = nn.Linear(dim, 2 * dim)
        
        # GLU activation
        self.glu = nn.GLU(dim=-1)
        
        # Depthwise conv
        self.depthwise_conv = nn.Conv1d(
            dim, dim, kernel_size,
            padding=kernel_size // 2,
            groups=dim
        )
        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = nn.SiLU()  # Swish activation
        
        # Pointwise conv
        self.pointwise_conv2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: [B, T, D]
        x = self.norm(x)
        
        # Pointwise conv + GLU
        x = self.pointwise_conv1(x)  # [B, T, 2D]
        x = self.glu(x)  # [B, T, D]
        
        # Depthwise conv
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = x.transpose(1, 2)  # [B, T, D]
        
        # Pointwise conv
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        return x


class ModalFusionLayer(nn.Module):
    """模態融合層：使用交叉注意力融合圖像和文本信息"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        # 交叉注意力用於融合
        self.cross_attn_img = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_txt = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 門控機制
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Sigmoid()
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, sequence, image_feat, text_feat):
        B, T, D = sequence.shape
        
        # 擴展全局特徵
        image_feat = image_feat.unsqueeze(1)  # [B, 1, D]
        text_feat = text_feat.unsqueeze(1)    # [B, 1, D]
        
        # 交叉注意力
        seq_norm = self.norm1(sequence)
        img_attn, _ = self.cross_attn_img(seq_norm, image_feat, image_feat)
        txt_attn, _ = self.cross_attn_txt(seq_norm, text_feat, text_feat)
        
        # 門控融合
        gate_input = torch.cat([
            sequence,
            img_attn,
            txt_attn
        ], dim=-1)  # [B, T, 3D]
        
        gate = self.gate(gate_input)  # [B, T, D]
        
        # 加權組合
        fused = sequence + gate * (img_attn + txt_attn)
        fused = self.norm2(fused)
        
        return fused


class EmbeddingConditioner(nn.Module):
    """主條件編碼器類"""
    def __init__(
        self,
        input_dim=2048,
        output_dim=1024,
        projector_type="sequence",  # 默認使用序列投影器
        seq_len=77,
        hidden_dim=1536,
        num_layers=3,
        num_heads=8,
        dropout=0.1,
        freeze=False,
        **kwargs
    ):
        super().__init__()
        
        # 選擇投影器類型
        if projector_type == "simple":
            self.projector = SimpleProjector(input_dim, output_dim, dropout)
            self.use_sequence = False
        elif projector_type == "sequence":
            self.projector = SequenceProjector(input_dim, output_dim, seq_len, dropout)
            self.use_sequence = True
        elif projector_type == "transformer":
            self.projector = TransformerProjector(
                input_dim, output_dim, seq_len, num_layers, num_heads, dropout
            )
            self.use_sequence = True
        elif projector_type == "hybrid":
            self.projector = HybridProjector(input_dim, output_dim, seq_len, dropout)
            self.use_sequence = True
        elif projector_type == "attention":
            self.projector = AttentionProjector(
                input_dim, output_dim, seq_len, num_heads, dropout
            )
            self.use_sequence = True
        elif projector_type == "conv":
            self.projector = ConvProjector(input_dim, output_dim, seq_len, dropout)
            self.use_sequence = True
        elif projector_type == "audio_aware":
            self.projector = AudioAwareProjector(
                input_dim, output_dim, seq_len, num_layers, num_heads, dropout
            )
            self.use_sequence = True
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")
        
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.seq_len = seq_len
        
        if freeze:
            self.freeze()
            
        print(f"{self.__class__.__name__} with {projector_type} projector")
        print(f"Number of parameters: {count_params(self) * 1e-6:.2f}M")
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def encode(self, embedding):
        """
        embedding: [B, 2048] 預計算的 embeddings
        return: [B, 77, 1024] 序列化的條件
        """
        if isinstance(embedding, list):
            embedding = torch.stack([torch.FloatTensor(e) for e in embedding])
        
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.FloatTensor(embedding)

        if embedding.device != next(self.parameters()).device:
            embedding = embedding.to(next(self.parameters()).device)
        
        # 使用投影器
        if self.use_sequence:
            # 投影器直接輸出序列
            return self.projector(embedding)
        else:
            # SimpleProjector需要手動擴展（保留用於對比）
            projected = self.projector(embedding)
            # 警告：這種方式會導致所有位置相同！
            projected = projected.unsqueeze(1).repeat(1, self.seq_len, 1)
            return projected
