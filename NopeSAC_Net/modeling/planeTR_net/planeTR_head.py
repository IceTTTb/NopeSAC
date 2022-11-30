import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.registry import Registry

from ..transformer.position_encoding import PositionEmbeddingSine
from ..transformer.transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder

__all__ = ["PlaneTRHead", "SEM_SEG_HEADS_REGISTRY", "build_planeTR_head"]
SEM_SEG_HEADS_REGISTRY = Registry("SEM_SEG_HEADS")

def build_planeTR_head(cfg, shape):
    name = cfg.MODEL.SEM_SEG_HEAD.NAME
    return SEM_SEG_HEADS_REGISTRY.get(name)(cfg, shape)

@SEM_SEG_HEADS_REGISTRY.register()
class PlaneTRHead(nn.Module):
    _version = 2
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    def __init__(
        self, cfg, input_shape: Dict[str, ShapeSpec]
    ):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.input_shape = input_shape
        self.backbone_channels = []
        for k, v in input_shape.items():
            self.backbone_channels.append(v.channels)
        self.param_on = cfg.MODEL.SEM_SEG_HEAD.PARAM_ON
        self.center_on = cfg.MODEL.SEM_SEG_HEAD.CENTER_ON
        self.depth_on = cfg.MODEL.DEPTH_ON
        self.return_inter = cfg.MODEL.SEM_SEG_HEAD.DEEP_SUPERVISION
        self.deep_supervision = cfg.MODEL.SEM_SEG_HEAD.DEEP_SUPERVISION
        self.plane_embedding_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        self.context_channels = self.backbone_channels[-1]
        self.hidden_dim = cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIM
        self.num_queries = cfg.MODEL.SEM_SEG_HEAD.NUM_OBJECT_QUERIES
        self.nheads = cfg.MODEL.SEM_SEG_HEAD.NHEADS
        self.enc_layers = cfg.MODEL.SEM_SEG_HEAD.ENC_LAYERS
        self.dec_layers = cfg.MODEL.SEM_SEG_HEAD.DEC_LAYERS
        self.channel = 256

        self.position_embedding = PositionEmbeddingSine(self.hidden_dim // 2, normalize=True)

        # context projection layer
        self.input_proj = nn.Conv2d(self.context_channels, self.hidden_dim, kernel_size=1)
        # context self-attention layer
        ca_layer = TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.nheads, dim_feedforward=1024,
                                           dropout=0.1, activation="relu", normalize_before=False)
        context_ca_norm = nn.LayerNorm(self.hidden_dim)
        self.context_SA = TransformerEncoder(ca_layer, num_layers=self.enc_layers, norm=context_ca_norm)
        # plane query
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        # context to plane decoder
        plane_decoder_norm = nn.LayerNorm(self.hidden_dim)
        decoder_layer = TransformerDecoderLayer(d_model=self.hidden_dim, nhead=self.nheads, dim_feedforward=1024,
                                                dropout=0.1, activation="relu", normalize_before=True)
        self.context2plane_decoder = TransformerDecoder(decoder_layer, num_layers=self.dec_layers, norm=plane_decoder_norm,
                                                        return_intermediate=self.return_inter)

        # -------------------------------------------------------------------------------------------------
        # pixel decoder
        self.top_down = top_down(cfg, self.backbone_channels, self.channel, m_dim=self.hidden_dim,
                                 double_upsample=False)
        # plane embedding
        self.plane_embedding = MLP(self.hidden_dim, self.hidden_dim, self.plane_embedding_dim, 3)
        self.pixel_embedding = nn.Conv2d(self.channel, self.plane_embedding_dim, (1, 1), padding=0)
        # plane / non-plane classifier
        self.plane_prob = nn.Linear(self.hidden_dim, self.num_classes + 1)
        # plane mask prob
        # plane param
        if self.param_on:
            # instance-level plane 3D parameters
            self.plane_param = MLP(self.hidden_dim, self.hidden_dim, 3, 3)
        # plane center
        if self.center_on:
            self.plane_center = MLP(self.hidden_dim, self.hidden_dim, 2, 3)
            self.pixel_plane_center = nn.Conv2d(self.channel, 2, (1, 1), padding=0)
        # depth
        if self.depth_on:
            self.top_down_depth = top_down(cfg, self.backbone_channels, self.channel, m_dim=self.hidden_dim,
                                           double_upsample=False)
            self.depth = nn.Conv2d(self.channel, 1, (1, 1), padding=0)

    def forward(self, features):
        # c1-res2: 256, 120, 160
        # c2-res3: 512, 60, 80
        # c3-res4: 1024, 30, 40
        # c4-res5: 2048, 15, 20 (TR-En input)

        c1, c2, c3, c4 = features['res2'], features['res3'], features['res4'], features['res5']

        # get feat and pos emb of c4
        c4_pos_map = self.position_embedding(c4)  # b, hidden_dim, h, w
        c4_feat_map = self.input_proj(c4)  # b, hidden_dim, h, w
        bs, dc, hc, wc = c4_feat_map.shape
        c4_feat_seq = c4_feat_map.flatten(2).permute(2, 0, 1)  # hw, b, hidden_dim
        c4_pos_seq = c4_pos_map.flatten(2).permute(2, 0, 1)  # hw, b, hidden_dim

        # conduct self-attention on c4
        c4_feat_map_after_sa = self.context_SA(c4_feat_seq, pos=c4_pos_seq)

        # transformer decoder
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        hs = self.context2plane_decoder(tgt=tgt, memory=c4_feat_map_after_sa, pos=c4_pos_seq,
                                        query_pos=query_embed)  # dec_layers, num_queries, b, hidden_dim
        hs = hs.transpose(1, 2)  # dec_layers, b, num_queries, hidden_dim
        hs = hs[-3:]

        # pixel decoder
        c4_feat_map_after_sa = c4_feat_map_after_sa.permute(1, 2, 0).view(bs, self.hidden_dim, hc, wc)
        p_context = self.top_down((c1, c2, c3, c4), c4_feat_map_after_sa)  # p0: b, c, 120, 160

        # --------------------------------------------------------- plane instance decoder
        # mask decoder
        plane_embedding = self.plane_embedding(hs)  # l, b, num_queries, c
        pixel_embedding = self.pixel_embedding(p_context)  # b, c, 120, 160
        pixel_masks_logits = torch.einsum("lbqc,bchw->lbqhw", plane_embedding,
                                          pixel_embedding)  # l, b, num_queries, 120, 160
        # plane classifier
        plane_logits = self.plane_prob(hs)  # l, b, num_queries, 2

        if self.param_on:
            # plane parameters
            plane_param = self.plane_param(hs)  # l, b, num_queries, 3
        if self.center_on:
            plane_center = self.plane_center(hs)  # l, b, num_queries, 2
            plane_center = torch.sigmoid(plane_center)
            pixel_center = self.pixel_plane_center(p_context)  # b, 2, h, w
            pixel_center = torch.sigmoid(pixel_center)  # 0~1
        if self.depth_on:
            p_depth = self.top_down_depth((c1, c2, c3, c4), c4_feat_map_after_sa)
            pixel_depth = self.depth(p_depth)  # b, 1, h, w

        # ------------------------------------------------------output
        output = {'pred_logits': plane_logits[-1],
                  'pred_mask_logits': pixel_masks_logits[-1]
                  }
        if self.param_on:
            output['pred_params'] = plane_param[-1]
        if self.center_on:
            output['pixel_centers'] = pixel_center
            output['pred_centers'] = plane_center[-1]
        if self.depth_on:
            output['pixel_depth'] = pixel_depth

        if self.deep_supervision and self.training:
            aux_outputs = []
            for i in range(plane_logits.shape[0] - 1):
                aux = {'pred_logits': plane_logits[i],
                       'pred_mask_logits': pixel_masks_logits[i]
                       }
                if self.param_on:
                    aux['pred_params'] = plane_param[i]
                if self.center_on:
                    aux['pred_centers'] = plane_center[i]
                aux_outputs.append(aux)
            output['aux_outputs'] = aux_outputs

        return output, hs[-1]

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def conv_bn_relu(in_dim, out_dim, k=1, pad=0):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, (k, k), padding=pad, bias=False),
        nn.BatchNorm2d(out_dim),
        # nn.SyncBatchNorm(out_dim),
        nn.ReLU(inplace=True)
    )


class top_down(nn.Module):
    def __init__(self, cfg, in_channels=[], channel=64, m_dim=256, double_upsample=False):
        super(top_down, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.double_upsample = double_upsample

        # top down
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        if double_upsample:
            self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.up_conv3 = conv_bn_relu(channel, channel, 1)
        self.up_conv2 = conv_bn_relu(channel, channel, 1)
        self.up_conv1 = conv_bn_relu(channel, channel, 1)

        # lateral
        self.c4_conv = conv_bn_relu(in_channels[3], channel, 1)
        self.c3_conv = conv_bn_relu(in_channels[2], channel, 1)
        self.c2_conv = conv_bn_relu(in_channels[1], channel, 1)
        self.c1_conv = conv_bn_relu(in_channels[0], channel, 1)

        self.m_conv_dict = nn.ModuleDict({})
        self.m_conv_dict['m4'] = conv_bn_relu(m_dim, channel)

    def forward(self, x, memory):
        c1, c2, c3, c4 = x

        p4 = self.c4_conv(c4) + self.m_conv_dict['m4'](memory)

        p3 = self.up_conv3(self.upsample(p4)) + self.c3_conv(c3)

        p2 = self.up_conv2(self.upsample(p3)) + self.c2_conv(c2)

        p1 = self.up_conv1(self.upsample(p2)) + self.c1_conv(c1)

        return p1
