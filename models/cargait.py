import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from ..base_model import BaseModel
from ..modules import (
    SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper,
    SeparateFCs, SeparateBNNecks, conv1x1, conv3x3,
    BasicBlock2D, BasicBlockP3D, Graph
)


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.weight * grad_output, None


def grad_reverse(x, weight=1.0):
    return GradientReverse.apply(x, weight)


class PoseGraphBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, temporal_kernel_size=5):
        super().__init__()
        if not torch.is_tensor(A):
            A = torch.tensor(A, dtype=torch.float32)
        A = A.mean(dim=0)
        degree = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
        self.register_buffer('A_norm', A / degree)

        self.spatial = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.spatial_bn = nn.BatchNorm2d(out_channels)
        padding = (temporal_kernel_size - 1) // 2
        self.temporal = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(temporal_kernel_size, 1),
            padding=(padding, 0), bias=False
        )
        self.temporal_bn = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        n, c, t, v = x.shape
        residual = x if self.residual is None else self.residual(x)

        x_perm = x.permute(0, 2, 3, 1).reshape(n * t, v, c)
        x_spatial = torch.bmm(
            self.A_norm.unsqueeze(0).expand(n * t, -1, -1), x_perm
        )
        x_spatial = x_spatial.permute(0, 2, 1).reshape(n, t, c, v)
        x_spatial = x_spatial.permute(0, 2, 1, 3).contiguous()

        out = self.spatial(x_spatial)
        out = self.relu(self.spatial_bn(out))
        out = self.temporal(out)
        out = self.temporal_bn(out)
        return self.relu(out + residual)


class PoseMotionEncoder(nn.Module):
    def __init__(self, out_channels=48, joint_format='coco', hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [48, 96, 96]
        graph = Graph(joint_format=joint_format, max_hop=2)
        self.num_joints = graph.A.shape[1]
        self.data_bn = nn.BatchNorm1d(self.num_joints * 5)
        self.blocks = nn.ModuleList()
        in_channels = 5
        for ch in hidden_channels:
            self.blocks.append(PoseGraphBlock(in_channels, ch, graph.A))
            in_channels = ch
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_channels[-1], out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(inplace=True)
        )

    def _normalize_pose(self, pose):
        coord = pose[..., :2]
        conf = pose[..., 2:3] if pose.size(-1) > 2 else torch.ones_like(coord[..., :1])
        conf = conf.clamp(0.0, 1.0)

        if pose.size(2) >= 17:
            center = (coord[:, :, 11:12] + coord[:, :, 12:13]) * 0.5
            shoulder = (coord[:, :, 5:6] - coord[:, :, 6:7]).norm(dim=-1, keepdim=True)
            hip = (coord[:, :, 11:12] - coord[:, :, 12:13]).norm(dim=-1, keepdim=True)
            scale = torch.maximum(shoulder, hip).clamp(min=1.0)
        else:
            valid = (conf > 0).float()
            center = (coord * valid).sum(dim=2, keepdim=True) / valid.sum(dim=2, keepdim=True).clamp(min=1.0)
            scale = coord.std(dim=2, keepdim=True).mean(dim=-1, keepdim=True).clamp(min=1.0)

        norm_coord = (coord - center) / scale
        velocity = torch.zeros_like(norm_coord)
        velocity[:, 1:] = norm_coord[:, 1:] - norm_coord[:, :-1]
        return torch.cat([norm_coord, velocity, conf], dim=-1)

    def forward(self, pose):
        pose = pose.float()
        if pose.dim() != 4:
            raise ValueError('CAR-Gait expects pose shape [B, T, V, C], got %s' % (list(pose.shape),))
        pose = self._normalize_pose(pose)
        n, t, v, c = pose.shape
        x = pose.permute(0, 2, 3, 1).reshape(n, v * c, t)
        x = self.data_bn(x)
        x = x.reshape(n, v, c, t).permute(0, 2, 3, 1).contiguous()
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=-1).transpose(1, 2).contiguous()
        return self.out_proj(x)


class ConditionAdaptiveFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.modulator = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 2, channels * 2)
        )
        self.pose_proj = nn.Linear(channels, channels)
        self.gate = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, sil_feat, pose_feat):
        n, c, t, h, w = sil_feat.shape
        if pose_feat.size(1) != t:
            pose_feat = F.interpolate(
                pose_feat.transpose(1, 2), size=t, mode='linear',
                align_corners=False
            ).transpose(1, 2).contiguous()

        gamma_beta = self.modulator(pose_feat).transpose(1, 2).reshape(n, c * 2, t, 1, 1)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        modulated = sil_feat * (1.0 + torch.tanh(gamma)) + torch.tanh(beta)

        pose_map = self.pose_proj(pose_feat).transpose(1, 2).reshape(n, c, t, 1, 1)
        pose_map = pose_map.expand(-1, -1, -1, h, w)
        gate = self.gate(torch.cat([modulated, pose_map], dim=1))
        return modulated + gate * pose_map


class CAR_Gait(BaseModel):
    def build_network(self, model_cfg):
        backbone_cfg = model_cfg['Backbone']
        self.channels = backbone_cfg.get('channels', [48, 96, 192, 384])
        self.layers = backbone_cfg.get('layers', [1, 1, 1, 1])
        self.inference_use_emb2 = model_cfg.get('use_emb2', False)

        self.condition_keys = model_cfg.get(
            'condition_keys',
            ['U0_D0', 'U0_D0_BG', 'U1_D0', 'U1_D1', 'U2_D2', 'U3_D3', 'U0_D3']
        )
        self.condition_to_idx = {k: i for i, k in enumerate(self.condition_keys)}
        self.num_conditions = len(self.condition_keys)
        loss_weights = model_cfg.get('loss_weights', {})
        self.lambda_condition_adv = float(loss_weights.get('condition_adv', 0.2))
        self.lambda_state_consistency = float(loss_weights.get('state_consistency', 0.2))
        self.lambda_pose_sil = float(loss_weights.get('pose_sil', 0.05))
        self.aux_warmup_iter = int(model_cfg.get('aux_warmup_iter', 20000))

        self.inplanes = self.channels[0]
        self.layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(1, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))
        self.layer1 = SetBlockWrapper(
            self._make_layer(BasicBlock2D, self.channels[0], [1, 1], self.layers[0], mode='2d')
        )

        self.pose_encoder = PoseMotionEncoder(
            out_channels=self.channels[0],
            joint_format=model_cfg.get('joint_format', 'coco'),
            hidden_channels=model_cfg.get('PoseEncoder', {}).get('hidden_channels', [48, 96, 96])
        )
        self.fusion = ConditionAdaptiveFusion(self.channels[0])

        self.layer2 = self._make_layer(BasicBlockP3D, self.channels[1], [2, 2], self.layers[1], mode='p3d')
        self.layer3 = self._make_layer(BasicBlockP3D, self.channels[2], [2, 2], self.layers[2], mode='p3d')
        self.layer4 = self._make_layer(BasicBlockP3D, self.channels[3], [1, 1], self.layers[3], mode='p3d')

        parts_num = model_cfg.get('parts_num', 16)
        fc_out = model_cfg.get('embedding_channels', self.channels[2])
        self.FCs = SeparateFCs(parts_num, self.channels[3], fc_out)
        self.BNNecks = SeparateBNNecks(
            parts_num, fc_out, class_num=model_cfg['SeparateBNNecks']['class_num']
        )
        self.condition_head = nn.Sequential(
            nn.Linear(fc_out, fc_out),
            nn.BatchNorm1d(fc_out),
            nn.ReLU(inplace=True),
            nn.Linear(fc_out, self.num_conditions)
        )
        self.pose_sil_proj = nn.Linear(self.channels[0], self.channels[0])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[parts_num])

    def _make_layer(self, block, planes, stride, blocks_num, mode='p3d'):
        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == '2d':
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                    nn.BatchNorm2d(planes * block.expansion)
                )
            elif mode == 'p3d':
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes, planes * block.expansion,
                        kernel_size=[1, 1, 1], stride=[1, *stride],
                        padding=[0, 0, 0], bias=False
                    ),
                    nn.BatchNorm3d(planes * block.expansion)
                )
            else:
                raise TypeError('Unsupported block mode: %s' % mode)
        else:
            downsample = None

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks_num):
            layers.append(block(self.inplanes, planes, stride=[1, 1]))
        return nn.Sequential(*layers)

    def _format_silhouette(self, sil):
        if sil.dim() == 4:
            return sil.unsqueeze(1).float()
        if sil.dim() == 5:
            if sil.size(1) == 1:
                return sil.float()
            return sil.transpose(1, 2).contiguous().float()
        raise ValueError('Unsupported silhouette shape: %s' % (list(sil.shape),))

    def _condition_labels(self, typs, device):
        labels = []
        for typ in typs:
            typ = str(typ)
            condition = typ if typ in self.condition_to_idx else typ.split('-', 1)[0]
            labels.append(self.condition_to_idx.get(condition, 0))
        return torch.tensor(labels, dtype=torch.long, device=device)

    def _aux_scale(self):
        if self.aux_warmup_iter <= 0:
            return 1.0
        return min(float(max(self.iteration, 1)) / float(self.aux_warmup_iter), 1.0)

    def _supervised_cross_state_loss(self, feat, labs, conds):
        feat = F.normalize(feat, dim=1)
        logits = torch.matmul(feat, feat.t()) / 0.07
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        logits_mask = torch.ones_like(logits)
        logits_mask.fill_diagonal_(0.0)

        same_id = labs.view(-1, 1).eq(labs.view(1, -1))
        diff_cond = conds.view(-1, 1).ne(conds.view(1, -1))
        pos_mask = (same_id & diff_cond).float() * logits_mask
        fallback_mask = same_id.float() * logits_mask
        has_pos = pos_mask.sum(dim=1, keepdim=True) > 0
        pos_mask = torch.where(has_pos, pos_mask, fallback_mask)

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))
        pos_count = pos_mask.sum(dim=1).clamp(min=1.0)
        loss = -(pos_mask * log_prob).sum(dim=1) / pos_count
        valid = pos_mask.sum(dim=1) > 0
        if valid.any():
            return loss[valid].mean()
        return feat.new_tensor(0.0)

    def forward(self, inputs):
        ipts, labs, typs, _, seqL = inputs
        if isinstance(ipts, (list, tuple)) and len(ipts) >= 2:
            pose = ipts[0]
            sil = ipts[1]
        else:
            pose = None
            sil = ipts[0] if isinstance(ipts, (list, tuple)) else ipts

        sils = self._format_silhouette(sil)
        out0 = self.layer0(sils)
        out1 = self.layer1(out0)

        if pose is not None:
            pose_feat = self.pose_encoder(pose)
        else:
            pose_feat = out1.mean(dim=[3, 4]).transpose(1, 2).contiguous().detach()

        sil_low_seq = out1.mean(dim=[3, 4]).transpose(1, 2).contiguous()
        fused = self.fusion(out1, pose_feat)
        out2 = self.layer2(fused)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        outs = self.TP(out4, seqL, options={"dim": 2})[0]
        feat = self.HPP(outs)
        embed_1 = self.FCs(feat)
        embed_2, logits = self.BNNecks(embed_1)
        embed = embed_2 if self.inference_use_emb2 else embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils[:, :, :1] * 255., 'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': embed
            }
        }

        if self.training and typs is not None:
            aux_scale = self._aux_scale()
            emb_vec = embed_1.mean(dim=2)
            conds = self._condition_labels(typs, emb_vec.device)

            adv_logits = self.condition_head(grad_reverse(emb_vec, aux_scale))
            condition_adv = F.cross_entropy(adv_logits.float(), conds)
            state_consistency = self._supervised_cross_state_loss(emb_vec.float(), labs, conds)

            pose_global = F.normalize(pose_feat.mean(dim=1).float(), dim=1)
            sil_global = F.normalize(self.pose_sil_proj(sil_low_seq.mean(dim=1)).float(), dim=1)
            pose_sil = 1.0 - F.cosine_similarity(pose_global, sil_global, dim=1).mean()

            retval['training_feat']['condition_adv'] = (
                aux_scale * self.lambda_condition_adv * condition_adv
            )
            retval['training_feat']['state_consistency'] = (
                aux_scale * self.lambda_state_consistency * state_consistency
            )
            retval['training_feat']['pose_sil_consistency'] = (
                aux_scale * self.lambda_pose_sil * pose_sil
            )
            retval['visual_summary']['scalar/cargait/aux_scale'] = (
                emb_vec.new_tensor(aux_scale).detach()
            )
            retval['visual_summary']['scalar/cargait/condition_acc'] = (
                (adv_logits.detach().argmax(dim=1) == conds).float().mean()
            )

        return retval
