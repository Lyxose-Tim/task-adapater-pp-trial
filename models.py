import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod
import math
import clip
from einops import rearrange
from module_adapter import clip_vit_base_patch16_adapter
from module_sem_adapter import clip_encode_text_adapter
from utils import *

class MetaTemplate(nn.Module):
    def __init__(self, n_way, n_support, n_query):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = n_query
    
    @abstractmethod
    def forward(self,x):
        pass
    
    def distribute_backbone(self,devices):
        self.feature.cuda(0)
        self.feature = torch.nn.DataParallel(self.feature,device_ids = devices)
    

params = argparse.Namespace(**read_yaml())
class TaskAdapter(MetaTemplate):
    def __init__(self, n_way, n_support, n_query, dataset,adapter_depth,text_depth):
        super().__init__(n_way, n_support, n_query)


        if dataset == 'hmdb51':
            self.cls_txt = clip.tokenize(hmdb_cls).cuda()
            self.cls_name = hmdb_c
            self.entxt = read_yml(dataset)#content
        elif dataset == 'ucf101':
            self.cls_txt = clip.tokenize(ucf_cls).cuda()
            self.cls_name = ucf_c
            self.entxt = read_yml(dataset)#content
        elif dataset == 'kinetics':
            self.cls_txt = clip.tokenize(kinetics_cls).cuda()
            self.cls_name = kinetics_c
            self.entxt = read_yml(dataset)#content
        elif 'something' in dataset:
            self.cls_txt = clip.tokenize(smsm_cls).cuda()
            self.cls_name = smsm_c
            self.entxt = read_yml(dataset)#content
        
        self.feature = clip_vit_base_patch16_adapter(embed_dim=512, adapter_layers=adapter_depth)
        self.text = clip_encode_text_adapter(adapter_layer=text_depth)

        self.ca=CrossAttention()

        # 创新点 1 · OT 软阶段分配开关（手册 §1.3；从全局 config 读，缺省=window 即 B0）
        self.align_mode = getattr(params, 'align_mode', 'window')   # window|ot
        self.ot_eps = float(getattr(params, 'ot_eps', 0.05))
        self.ot_lam = float(getattr(params, 'ot_lam', 0.0))
        _rho = getattr(params, 'ot_rho', None)                      # None=平衡
        self.ot_rho = None if _rho in (None, 'none', 'None', '') else float(_rho)
        self.ot_iters = int(getattr(params, 'ot_iters', 30))
        self.ot_weight = getattr(params, 'ot_weight', 'mass')       # mass|uniform
        self.frame_source = getattr(params, 'frame_source', 'ca')   # ca|raw(Option B)
        self.ca_residual = getattr(params, 'ca_residual', 'next')   # next(开源现状)|prev(式13)
    
    
    def episode_visual(self, x):
        # 视觉前向（与 label/文本无关）。诊断时每 episode 只做一次并复用
        # z_query/z_proto/q_aft_tm（手册 §2.3）。返回值形状：
        #   z_query [N_Q, 8, d] · z_proto [n_way, 8, d] · q_aft_tm [7, N_Q, d]
        N,C,H,W = x.shape
        x = x.reshape(self.n_way*(self.n_query+self.n_support), 8 , C, H, W)
        x = x.permute(0, 2, 1, 3, 4) # B C T H W

        x = x.reshape(self.n_way, (self.n_query+self.n_support), C , 8, H, W)

        support_images = x[:, :self.n_support].reshape(self.n_way*self.n_support, C, 8, H, W)
        target_images = x[:,self.n_support:].reshape(self.n_way*self.n_query, C, 8, H, W)

        z_support  = self.feature(support_images).reshape(self.n_way*self.n_support, 8, -1)
        z_query = self.feature(target_images).reshape(self.n_way*self.n_query, 8, -1)

        z_proto = z_support.reshape(self.n_way, self.n_support, 8, -1).mean(1)
        z_query = z_query.reshape(self.n_way*self.n_query, 8, -1 )

        q_reshape=z_query.permute(1,0,2)
        q_aft_tm=[]

        for frame in range(7):
            attended = self.ca(q_reshape[frame],q_reshape[frame+1],q_reshape[frame+1])
            # P3 残差：next=开源现状（加 F^{i+1}=q_reshape[frame+1]）；prev=式(13)（加 q_reshape[frame]）
            # getattr 默认 'next'：经 __new__ 构造的测试模型（未跑 __init__）仍走 B0 行为
            residual = q_reshape[frame] if getattr(self, 'ca_residual', 'next') == 'prev' else q_reshape[frame+1]
            q_aft_tm.append(attended + residual)

        q_aft_tm=torch.stack(q_aft_tm,dim=0)
        return z_query, z_proto, q_aft_tm

    def _encode_stage_text(self, label_idx, permutation=None):
        # 文本前向：按 label 逐类编码 K 个子动作 → enh_embedding [K, n_way, D]。
        # permutation（长度 K 的下标序）先重排每类 sub_act_en_li。逐类调用是
        # O-MSA 语义要求（R-02），不可整 episode 合批。
        label_name = [self.cls_name[i] for i in label_idx]
        enhtxt=[]
        tmp_prompt = []
        for i in label_name:
            subs = self.entxt[i]['sub_act_en_li']
            if permutation is not None:
                subs = [subs[p] for p in permutation]
            for j in subs:
                one_prompt = f'A video of action about {i}: {j}'
                tmp_prompt.append(one_prompt)
            enhtxt.append(tmp_prompt)
            tmp_prompt=[]

        enh_embedding = []
        for i in range(self.n_way):
            enh_embedding.append(self.text(clip.tokenize(enhtxt[i]).to('cuda')))
        return torch.stack(enh_embedding,dim=0).permute(1,0,2)   # [K, n_way, D]

    def _fixed_window_score(self, enh_embedding, q_aft_tm):
        # 阶段-窗口 cos（式(16) 口径）：阶段 i 对齐窗口 [2i,2i+1,2i+2]，/9。
        cos_score = 0
        for i in range(3):
            cos_score += (cosine_similarity(enh_embedding[i],q_aft_tm[2*i])+cosine_similarity(enh_embedding[i],q_aft_tm[2*i+1])+cosine_similarity(enh_embedding[i],q_aft_tm[2*i+2]))
        cos_score = cos_score/9
        # P2: cos_score 朝向 [n_way, N_Q]，转置为 (查询行, 类列) 与视觉分一致
        return cos_score.transpose(-2, -1)

    def _ot_frames_and_text(self, enh_embedding, q_aft_tm, z_query=None):
        # 组装 OT 输入：F [NQ,T,D]（ca 7帧 / raw 8帧）、T_c [n_way,K,D]。fp32。
        if self.frame_source == 'raw':
            if z_query is None:
                raise ValueError("frame_source=raw 需要 z_query（Option B）")
            F_frames = z_query.float()                    # [NQ, 8, D]
        else:
            F_frames = q_aft_tm.permute(1, 0, 2).float()  # [NQ, 7, D]
        T_c = enh_embedding.permute(1, 0, 2).float()      # [n_way, K, D]
        return F_frames, T_c

    def _ot_stage_score(self, enh_embedding, q_aft_tm, z_query=None):
        # 创新点 1：用 OT 软阶段分配替换式(16) 固定窗口。手册 §1.3。
        import ot_align   # 延迟导入：window 模式（B0）永不触及 OT/fsar.ot
        F_frames, T_c = self._ot_frames_and_text(enh_embedding, q_aft_tm, z_query)
        S, _, _ = ot_align.ot_stage_scores(
            F_frames, T_c, eps=self.ot_eps, lam=self.ot_lam, rho=self.ot_rho,
            iters=self.ot_iters, weight=self.ot_weight)
        return S.to(q_aft_tm.dtype)                       # fp32 岛出口回半精度，接式(17)

    def ot_diagnostics(self, q_aft_tm, label_idx, z_query=None):
        # 无侵入插桩：返回 C0 正序 OT plan 的数值健康统计（π.detach()，
        # **不改分数/梯度/显存图**——本方法独立于打分路径，仅诊断调用）。
        import ot_align
        enh = self._encode_stage_text(label_idx, None)
        F_frames, T_c = self._ot_frames_and_text(enh, q_aft_tm, z_query)
        _, plan, _ = ot_align.ot_stage_scores(
            F_frames, T_c, eps=self.ot_eps, lam=self.ot_lam, rho=self.ot_rho,
            iters=self.ot_iters, weight=self.ot_weight)
        return ot_align.ot_plan_stats(plan)

    def semantic_scores(self, q_aft_tm, label_idx, permutation=None, z_query=None):
        # 语义分支打分。align_mode=window 走式(16) 固定窗口（逐位=B0）；
        # =ot 走创新点 1 的 OT 软阶段分配。`permutation` 仅重排每类子动作顺序
        # （只影响语义分支、不动视觉；两模式通用，供 3a 诊断施加）。
        enh = self._encode_stage_text(label_idx, permutation)
        # getattr 默认 'window'：__new__ 构造的测试模型（未跑 __init__）走 B0 固定窗口
        if getattr(self, 'align_mode', 'window') == 'ot':
            return self._ot_stage_score(enh, q_aft_tm, z_query)
        return self._fixed_window_score(enh, q_aft_tm)

    def order_margin_loss(self, q_aft_tm, label_idx, order_perms, margin):
        # 创新点 3b·顺序对比正则（手册 §3.1 margin 版 + §3.2 免费午餐）：
        #   L_order = mean_{query i, 排列 π} max(0, γ − (s_i(正序) − s_i(π)))
        # s_i = 查询 i 与其正确类文本的语义匹配分（cos_score 通路的对角）。
        # O-1 等变（真权重门控 max_abs=0.0）：排列文本特征 = 正序 enh 按 K 轴
        # index_select，**零额外文本前向**；梯度经"阶段-窗口匹配"塑造阶段特异性。
        enh = self._encode_stage_text(label_idx, None)            # 正序编码一次 [K, n_way, D]
        sem_id = self._fixed_window_score(enh, q_aft_tm)          # [N_Q, n_way]
        y = torch.arange(self.n_way, device=sem_id.device).repeat_interleave(self.n_query).view(-1, 1)
        s_pos = sem_id.gather(1, y).squeeze(1)                    # [N_Q] 正确类正序分
        losses = []
        for perm in order_perms:
            idx = torch.as_tensor(list(perm), dtype=torch.long, device=enh.device)
            sem_p = self._fixed_window_score(enh.index_select(0, idx), q_aft_tm)
            s_perm = sem_p.gather(1, y).squeeze(1)
            losses.append(F.relu(margin - (s_pos - s_perm)))     # [N_Q]
        return torch.stack(losses, dim=0).mean()

    def forward_train(self, x, label, order_perms=None, order_margin=0.1):
        # 训练用前向：视觉前向一次，供 CE（vis*sem）与 order 正则共用。
        # order_perms=None → order_loss 返回 None（等价 B0 训练）。
        label_idx = list(label[:,0].numpy())
        z_query, z_proto, q_aft_tm = self.episode_visual(x)
        vis_dists = self.visual_scores(z_query, z_proto)
        sem_dists = self.semantic_scores(q_aft_tm, label_idx, None, z_query=z_query)
        order_loss = None
        if order_perms is not None:
            order_loss = self.order_margin_loss(q_aft_tm, label_idx, order_perms, order_margin)
        return vis_dists, sem_dists, order_loss

    def visual_scores(self, z_query, z_proto):
        # 视觉分支：query/proto 帧均值的余弦，[N_Q, n_way]
        return cosine_similarity(z_query.mean(1), z_proto.mean(1)).squeeze()

    def forward(self, x, label, permutation=None):
        # 行为与重构前逐算子一致（B0 数值不变）；仅拆出可复用方法并加
        # 诊断用的可选 permutation 参数。默认 permutation=None 即原始正序。
        label_idx = list(label[:,0].numpy())
        z_query, z_proto, q_aft_tm = self.episode_visual(x)
        vis_dists = self.visual_scores(z_query, z_proto)
        sem_dists = self.semantic_scores(q_aft_tm, label_idx, permutation, z_query=z_query)

        n_q_total = self.n_way * self.n_query
        assert sem_dists.shape == (n_q_total, self.n_way), \
            f"sem_dists must be (N_Q, n_way)=({n_q_total},{self.n_way}), got {tuple(sem_dists.shape)}"
        assert vis_dists.shape == (n_q_total, self.n_way), \
            f"vis_dists must be (N_Q, n_way)=({n_q_total},{self.n_way}), got {tuple(vis_dists.shape)}"

        # P1: 返回 (视觉分, 语义分)，与 run.py 接收点命名一致
        return vis_dists, sem_dists


class CrossAttention(nn.Module):
    def __init__(self, dim=512, heads = 1, dim_head = 512, dropout = 0.1):
        super().__init__()
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.ln_1=nn.LayerNorm(dim) if params.dataset=='somethingotam'  else nn.Identity()
        self.attend = nn.Softmax(dim = -1)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q,k,v):
        
        dots = q@k.transpose(0,1) * self.scale

        attn = self.attend(dots)

        out = attn@v
        
        return self.ln_1(out)

