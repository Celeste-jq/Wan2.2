import logging
import torch
from torch import Tensor
import torch_npu
import torch.distributed as dist
import math
import os
from yunchang import LongContextAttention

try:
    from yunchang.kernels import AttnType
except ImportError:
    raise ImportError("Please install yunchang 0.6.0 or later")
from typing import Any

from mindiesd.layers.flash_attn.attention_forward import attention_forward

from ..distributed.parallel_mgr import get_sp_group
from ..distributed.comm import all_to_all_4D

logger = logging.getLogger(__name__)
MAX_TOKEN = 2147483647
local_rank = int(os.getenv('LOCAL_RANK', 0))
stream = torch.npu.Stream(torch.device(f"cuda:{local_rank}"))


class xFuserLongContextAttention(LongContextAttention):
    ring_impl_type_supported_kv_cache = ["basic"]

    def __init__(
        self,
        args: Any,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_pack_qkv: bool = False,
        use_kv_cache: bool = False,
        attn_type: AttnType = AttnType.FA,
        rainfusion_config=None,
        fa_quant=None,
    ) -> None:
        """
        Arguments:
            scatter_idx: int = 2, the scatter dimension index for Ulysses All2All
            gather_idx: int = 1, the gather dimension index for Ulysses All2All
            ring_impl_type: str = "basic", the ring implementation type, currently only support "basic"
            use_pack_qkv: bool = False, whether to use pack qkv in the input
            use_kv_cache: bool = False, whether to use kv cache in the attention layer, which is applied in PipeFusion.
        """
        super().__init__(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            ring_impl_type=ring_impl_type,
            use_pack_qkv=use_pack_qkv,
            attn_type=attn_type,
        )
        self.use_kv_cache = use_kv_cache
        if (
                use_kv_cache
                and ring_impl_type not in self.ring_impl_type_supported_kv_cache
        ):
            raise RuntimeError(
                f"ring_impl_type: {ring_impl_type} do not support SP kv cache."
            )
        self.world_size = dist.get_world_size()
        self.args = args
        self.video_size = ['480*832', '832*480', '480*720', '720*480']

        self.algo = int(os.getenv('ALGO', 0))

        if self.args.size in self.video_size:
            self.use_all_head = True
        else:
            self.use_all_head = False

        self.ulysses_pg = get_sp_group().ulysses_group
        self.ring_pg = get_sp_group().ring_group

        self.ulysess_world_size = dist.get_world_size(self.ulysses_pg)
        self.ring_world_size = get_sp_group().ring_world_size

        self.rainfusion_config = rainfusion_config
        self.rainfusion_fa = None
        if self.rainfusion_config is not None:
            if rainfusion_config["type"] == "v1":
                from wan.utils.rainfusion import Rainfusion
                self.rainfusion_fa = Rainfusion(
                    grid_size=rainfusion_config["grid_size"],
                    skip_timesteps=rainfusion_config["skip_timesteps"],
                    sparsity=rainfusion_config["sparsity"],
                )
            else:
                from wan.utils.rainfusion_blockwise import Rainfusion_blockwise
                self.rainfusion_fa_blockwise = Rainfusion_blockwise(
                    grid_size=rainfusion_config["grid_size"],
                    pool_size=128,
                    sparsity=rainfusion_config["sparsity"],
                    skip_timesteps=rainfusion_config["skip_timesteps"],
                    txt_len=0,
                )
        if fa_quant:
            self.fa_quant = fa_quant

        self.fa_alltoall_overlap = int(os.getenv('OVERLAP', 0))
        if self.ring_world_size > 1:
            self.fa_alltoall_overlap = False
        if self.fa_alltoall_overlap == True:
            if "ti2v" in self.args.task:
                event_nums = 24 // self.ulysess_world_size
            else:
                event_nums = 40 // self.ulysess_world_size
            self.current_stream = torch.npu.current_stream()
            self.stream2 = stream
            self.event = []
            self.event_begin = []
            for i in range(event_nums):
                self.event.append(torch.npu.Event())
                self.event_begin.append(torch.npu.Event())

    def forward(
        self,
        attn,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        seq_lens: int,
        *,
        joint_tensor_query=None,
        joint_tensor_key=None,
        joint_tensor_value=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        joint_strategy="none",
        scale=None,
        t_idx=-1,
        b_idx=-1
    ) -> Tensor:
        """forward

        Arguments:
            attn (Attention): the attention module
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args,
            joint_tensor_query: Tensor = None, a replicated tensor among processes appended to the front or rear of query, depends the joint_strategy
            joint_tensor_key: Tensor = None, a replicated tensor among processes appended to the front or rear of key, depends the joint_strategy
            joint_tensor_value: Tensor = None, a replicated tensor among processes appended to the front or rear of value, depends the joint_strategy,
            *args: the args same as flash_attn_interface
            joint_strategy: str = "none", the joint strategy for joint attention, currently only support "front" and "rear"

        Returns:
            * output (Tensor): context output
        """
        if self.fa_alltoall_overlap == True:
            query_layer_list = query.split(self.ulysess_world_size, dim=2)
            key_layer_list = key.split(self.ulysess_world_size, dim=2)
            value_layer_list = value.split(self.ulysess_world_size, dim=2)
            for_loop = len(query_layer_list)

            output_fa = []
            qkv_event = torch.npu.Event()
            qkv_event.record()
            q_lists, k_lists, v_lists = [], [], []

            with torch.npu.stream(self.current_stream):
                query_layer = all_to_all_4D(input_=query_layer_list[0], scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
                key_layer = all_to_all_4D(input_=key_layer_list[0], scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
                value_layer = all_to_all_4D(input_=value_layer_list[0], scatter_idx=2, gather_idx=1, group=self.ulysses_pg)

                ori_seqlen = query_layer.shape[1]
                need_pad = False

                if seq_lens is not None and seq_lens < ori_seqlen:
                    need_pad = True
                    # 照搬 else 分支逻辑：切分出 pad 部分，并将 query_layer 重赋值为有效部分
                    query_layer, query_pad = query_layer[:, :seq_lens, :, :], query_layer[:, seq_lens:, :, :]
                    key_layer, key_pad = key_layer[:, :seq_lens, :, :], key_layer[:, seq_lens:, :, :]
                    value_layer, value_pad = value_layer[:, :seq_lens, :, :], value_layer[:, seq_lens:, :, :]

                q_lists.append(query_layer)
                k_lists.append(key_layer)
                v_lists.append(value_layer)

                if self.algo == 0:
                    out = attention_forward(q_lists[0], k_lists[0], v_lists[0],
                                            opt_mode="manual", op_type="fused_attn_score", layout="BNSD")
                elif self.algo == 1:
                    out = attention_forward(q_lists[0], k_lists[0], v_lists[0],
                                            opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")
                elif self.algo == 3:
                    if hasattr(self, 'fa_quant'):
                        out = self.fa_quant(q_lists[0].transpose(1,2), k_lists[0].transpose(1,2), v_lists[0].transpose(1,2), layout="BNSD")
                    else:
                        scale = q_lists[0].shape[-1] ** -0.5
                        out = torch_npu.npu_fused_infer_attention_score(q_lists[0].transpose(1,2), k_lists[0].transpose(1,2), v_lists[0].transpose(1,2),
                            num_heads = 1, input_layout = "BNSD", scale = scale, pre_tokens=2147483647, next_tokens=2147483647)[0]
                    out = out.transpose(1,2)
                else:
                    raise ValueError(f"select flash attention algorithm only support 0, 1, 3, but got f{self.algo}")

                if need_pad:
                    out_pad = attention_forward(query_pad, key_pad, value_pad,
                                                opt_mode="manual", op_type="fused_attn_score", layout="BSND")
                    out = torch.cat([out, out_pad], dim=1)

                output_fa.append(out)
                self.event[0].record()

            with torch.npu.stream(self.stream2):
                for i in range(1, for_loop):
                    self.stream2.wait_event(qkv_event)
                    # B, S, 1, D
                    query_layer = all_to_all_4D(input_=query_layer_list[i], scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
                    key_layer = all_to_all_4D(input_=key_layer_list[i], scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
                    value_layer = all_to_all_4D(input_=value_layer_list[i], scatter_idx=2, gather_idx=1, group=self.ulysses_pg)

                    self.event[i].record(self.stream2)  # 记录qkv all2all的时间
                    q_lists.append(query_layer)
                    k_lists.append(key_layer)
                    v_lists.append(value_layer)

            for i in range(1, for_loop):
                self.current_stream.wait_event(self.event[i])
                query_layer = q_lists[i]
                key_layer = k_lists[i]
                value_layer = v_lists[i]

                ori_seqlen = query_layer.shape[1]
                need_pad = False
                if seq_lens is not None and seq_lens < ori_seqlen:
                    need_pad = True
                    # 照搬 else 分支逻辑：切分出 pad 部分，并将 query_layer 重赋值为有效部分
                    query_layer, query_pad = query_layer[:, :seq_lens, :, :], query_layer[:, seq_lens:, :, :]
                    key_layer, key_pad = key_layer[:, :seq_lens, :, :], key_layer[:, seq_lens:, :, :]
                    value_layer, value_pad = value_layer[:, :seq_lens, :, :], value_layer[:, seq_lens:, :, :]

                if self.algo == 0:
                    out = attention_forward(query_layer, key_layer, value_layer,
                                            opt_mode="manual", op_type="fused_attn_score", layout="BNSD")
                elif self.algo == 1:
                    out = attention_forward(query_layer, key_layer, value_layer,
                                            opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")
                elif self.algo == 3:
                    if hasattr(self, 'fa_quant'):
                        out = self.fa_quant(query_layer.transpose(1,2), key_layer.transpose(1,2), value_layer.transpose(1,2), layout="BNSD")
                    else:
                        scale = query_layer.shape[-1] ** -0.5
                        out = torch_npu.npu_fused_infer_attention_score(query_layer.transpose(1,2), key_layer.transpose(1,2), value_layer.transpose(1,2),
                            num_heads = 1, input_layout = "BNSD", scale = scale, pre_tokens=2147483647, next_tokens=2147483647)[0]
                    out = out.transpose(1,2)
                else:
                    raise ValueError(f"select flash attention algorithm only support 0, 1, 3, but got f{self.algo}")

                # 照搬 else 分支逻辑：如果有 pad，计算 pad 部分并拼接
                if need_pad:
                    out_pad = attention_forward(query_pad, key_pad, value_pad,
                                                opt_mode="manual", op_type="fused_attn_score", layout="BSND")
                    out = torch.cat([out, out_pad], dim=1)

                output_fa.append(out)
                self.event[i].record()

            output_res = []
            with torch.npu.stream(self.stream2):
                for i in range(for_loop):
                    self.stream2.wait_event(self.event[i])
                    # BSND gather 2 scatetrer 1
                    output_tmp = all_to_all_4D(input_=output_fa[i], scatter_idx=1, gather_idx=2, group=self.ulysses_pg)
                    self.event[i].record(self.stream2)
                    output_res.append(output_tmp)
            for i in range(for_loop):
                self.current_stream.wait_event(self.event[i])
            output = torch.cat(output_res, dim=2)

        else:
            query = all_to_all_4D(input_=query, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
            key = all_to_all_4D(input_=key, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
            value = all_to_all_4D(input_=value, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)

            if get_sp_group().ring_world_size > 1:
                ring_size = get_sp_group().ring_world_size
                b, s, n, d = key.shape
                k_full = torch.empty([ring_size, b, s, n, d], dtype=query.dtype, device=query.device)
                dist.all_gather_into_tensor(k_full, key, group=self.ring_pg)
                key = k_full.permute(1, 0, 2, 3, 4).reshape(b, -1, n, d)

                v_full = torch.empty([ring_size, b, s, n, d], dtype=query.dtype, device=query.device)
                dist.all_gather_into_tensor(v_full, value, group=self.ring_pg)
                value = v_full.permute(1, 0, 2, 3, 4).reshape(b, -1, n, d)

            ori_seqlen = query.shape[1]
            if seq_lens is not None and seq_lens < ori_seqlen:
                query_layer, query_pad = query[:, :seq_lens, :, :], query[:, seq_lens:, :, :]
                key_layer, key_pad = key[:, :seq_lens, :, :], key[:, seq_lens:, :, :]
                value_layer, value_pad = value[:, :seq_lens, :, :], value[:, seq_lens:, :, :]
            else:
                query_layer, key_layer, value_layer = query, key, value

            if self.rainfusion_config is not None:
                if self.rainfusion_config["type"] == "v1":
                    out = self.rainfusion_fa(
                        query_layer,
                        key_layer,
                        value_layer,
                        atten_mask_all=self.rainfusion_config["atten_mask_all"],
                        text_len=0,
                        t_idx=t_idx,
                    )
                else:
                    out, _ = self.rainfusion_fa_blockwise(
                       query_layer,
                       key_layer,
                       value_layer,
                       t_b_idx=[t_idx, b_idx],
                       base_blockmask=None,
                   )
            elif self.use_all_head:
                if self.algo == 0:
                    out = attention_forward(query_layer, key_layer, value_layer,
                                            opt_mode="manual", op_type="fused_attn_score", layout="BNSD")
                elif self.algo == 1:
                    out = attention_forward(query_layer, key_layer, value_layer,
                                            opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")
                elif self.algo == 3:
                    if hasattr(self, 'fa_quant'):
                        out = self.fa_quant(query_layer.transpose(1,2), key_layer.transpose(1,2), value_layer.transpose(1,2), layout="BNSD")
                    else:
                        scale = query_layer.shape[-1] ** -0.5
                        out = torch_npu.npu_fused_infer_attention_score(query_layer.transpose(1,2), key_layer.transpose(1,2), value_layer.transpose(1,2),
                            num_heads = query_layer.shape[2], input_layout = "BNSD", scale = scale, pre_tokens=2147483647, next_tokens=2147483647)[0]
                    out = out.transpose(1,2)
                else:
                    raise ValueError(f"select flash attention algorithm only support 0, 1, 3, but got f{self.algo}")
            else:
                query_layer_list = query_layer.split(1, dim=2)
                key_layer_list = key_layer.split(1, dim=2)
                value_layer_list = value_layer.split(1, dim=2)
                output = []
                for_loop = query_layer.shape[2]
                if scale is None:
                    scale = query.shape[-1] ** -0.5
                for i in range(for_loop):
                    if self.algo == 0:
                        out = attention_forward(query_layer_list[i], key_layer_list[i], value_layer_list[i],
                                                opt_mode="manual", op_type="fused_attn_score", layout="BNSD")
                    elif self.algo == 1:
                        out = attention_forward(query_layer_list[i], key_layer_list[i], value_layer_list[i],
                                                opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")
                    elif self.algo == 3:
                        if hasattr(self, 'fa_quant'):
                            out = self.fa_quant(query_layer_list[i].transpose(1,2), key_layer_list[i].transpose(1,2), value_layer_list[i].transpose(1,2), layout="BNSD")
                        else:
                            out = torch_npu.npu_fused_infer_attention_score(query_layer_list[i].transpose(1,2), key_layer_list[i].transpose(1,2), value_layer_list[i].transpose(1,2),
                                num_heads = 1, input_layout = "BNSD", scale = scale, pre_tokens=2147483647, next_tokens=2147483647)[0]
                        out = out.transpose(1,2)
                    else:
                        raise ValueError(f"select flash attention algorithm only support 0, 1, 3, but got f{self.algo}")

                    output.append(out)
                out = torch.cat(output, dim=2)

            if seq_lens is not None and seq_lens < ori_seqlen:
                out_pad = attention_forward(
                    query_pad, key_pad, value_pad,
                    opt_mode="manual", op_type="fused_attn_score", layout="BSND")
                out = torch.cat([out, out_pad], dim=1)

            if type(out) == tuple:
                context_layer, _, _ = out
            else:
                context_layer = out

            # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
            # scatter 1, gather 2
            output = all_to_all_4D(input_=context_layer, scatter_idx=1, gather_idx=2, group=self.ulysses_pg)

        return output


class QuantAllToAllAttention(xFuserLongContextAttention):
    """xFuserLongContextAttention with FP8 pre-quantization before Ulysses All-to-All.

    Activated when QUANT_ALLTOALL=1 and ALGO=3 and ring_world_size==1.
    Falls back to the BF16 baseline (super().forward) otherwise.
    """
    _rot_matrices: dict = {}

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.quant_alltoall = int(os.getenv('QUANT_ALLTOALL', 0))

    @classmethod
    def _get_rot(cls, head_dim, device, dtype):
        """Return (q_rot, k_rot) orthogonal matrices of shape [head_dim, head_dim].

        Q and K share the same rotation matrix R so that attention scores are preserved:
            Q' @ K'^T = (Q @ R) @ (K @ R)^T = Q @ R @ R^T @ K^T = Q @ K^T
        This keeps the attention numerically identical while smoothing the per-block
        value distribution for FP8 quantization (same principle as QuaRot/SpinQuant).

        Generated once with a fixed seed and cached at the class level, so the same
        matrix is used across all per-call instantiations of this class.
        """
        if head_dim not in cls._rot_matrices:
            gen = torch.Generator()
            gen.manual_seed(42)
            rot, _ = torch.linalg.qr(torch.randn(head_dim, head_dim, generator=gen))
            cls._rot_matrices[head_dim] = rot  # stored on CPU fp32
        rot = cls._rot_matrices[head_dim]
        rot = rot.to(device=device, dtype=dtype)
        return rot, rot

    def _fp8_attn(self, q_bnsd, k_bnsd, v_bnsd, out_dtype,
                  q_scale=None, k_scale=None, v_scale=None):
        """FP8 block-quantized attention (BNSD input/output in BSND).

        If scales are None, applies rotation matrices (like FP8RotateQuantFA) then
        quantizes q/k/v from BF16 on the spot (post-quant path).
        If scales are provided, uses them directly (pre-quant path, rotation already
        applied before all_to_all).
        """
        if q_scale is None:
            # Rotate Q and K before quantization to smooth the per-block distribution,
            # matching the FP8RotateQuantFA baseline.
            q_rot, k_rot = self._get_rot(q_bnsd.shape[-1], q_bnsd.device, q_bnsd.dtype)
            q_bnsd = torch.matmul(q_bnsd, q_rot)
            k_bnsd = torch.matmul(k_bnsd, k_rot)
            from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess
            q_bnsd, q_scale = fa_block_quant_preprocess(
                q_bnsd, block_size=128, dst_type=torch_npu.float8_e4m3fn, layout="BNSD")
            k_bnsd, k_scale = fa_block_quant_preprocess(
                k_bnsd, block_size=256, dst_type=torch_npu.float8_e4m3fn, layout="BNSD")
            v_bnsd, v_scale = fa_block_quant_preprocess(
                v_bnsd, block_size=256, dst_type=torch_npu.float8_e4m3fn, layout="BNSD")
        n_heads, s, head_dim = q_bnsd.shape[1], q_bnsd.shape[2], q_bnsd.shape[3]
        out = torch_npu.npu_fused_infer_attention_score_v2(
            q_bnsd, k_bnsd, v_bnsd,
            input_layout="BNSD",
            num_query_heads=n_heads,
            softmax_scale=1.0 / math.sqrt(head_dim),
            pre_tokens=MAX_TOKEN, next_tokens=MAX_TOKEN,
            query_quant_mode=7, key_quant_mode=7, value_quant_mode=7,
            dequant_scale_query=q_scale,
            dequant_scale_key=k_scale,
            dequant_scale_value=v_scale,
            out_dtype=out_dtype,
        )[0]
        out = out.transpose(1, 2)   # BNSD → BSND
        if out.shape[1] != s:
            out = out[:, :s, :, :]
        return out

    def _scale_all_to_all(self, scale):
        """All_to_all for FP8 block scales: [B, N_all, blocks_local, 1] → [B, N/P, blocks_full, 1].

        Reuses all_to_all_4D by temporarily treating the scale as a BSND-like tensor:
        transpose [B, N, blocks, 1] → [B, blocks, N, 1], scatter N / gather blocks, then transpose back.

        Used by the non-overlap branch where q_fp8 is all_to_all'd as a whole tensor (block head pattern).
        """
        t = scale.squeeze(-1).transpose(1, 2).unsqueeze(-1)
        t = all_to_all_4D(t, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
        return t.squeeze(-1).transpose(1, 2).unsqueeze(-1)

    def _chunk_scale_a2a(self, scale, chunk_idx, block_size, s_full):
        """All_to_all for one scale chunk matching the strided q_fp8 chunk split in the overlap branch.

        In the overlap branch, q_fp8 is split with `.split(P, dim=2)` (P = ulysess_world_size), so chunk i
        contains heads [i*P : (i+1)*P] and after all_to_all rank r gets global head i*P+r.
        This function distributes the matching scale slice using the SAME scatter/gather axes.

        scale:       [B, N, blocks_local, 1]
        chunk_idx:   i  (same as q_fp8 chunk index)
        block_size:  128 for Q, 256 for K/V
        s_full:      total sequence length after all_to_all (= S_local * P)

        Returns: [B, 1, ceil(s_full / block_size), 1] — scale for head i*P+r on rank r.
        """
        P = self.ulysess_world_size
        sc = scale[:, chunk_idx * P:(chunk_idx + 1) * P, :, :]  # [B, P, blocks_local, 1]
        # Reuse all_to_all_4D: treat dim-1 as "blocks" (to gather), dim-2 as "N=P" (to scatter)
        sc = sc.squeeze(-1).transpose(1, 2).unsqueeze(-1)        # [B, blocks_local, P, 1]
        sc = all_to_all_4D(sc, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
        sc = sc.squeeze(-1).transpose(1, 2).unsqueeze(-1)        # [B, 1, blocks_full, 1]
        return sc[:, :, :math.ceil(s_full / block_size), :]

    def forward(
        self,
        attn,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        seq_lens: int,
        *,
        joint_tensor_query=None,
        joint_tensor_key=None,
        joint_tensor_value=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        joint_strategy="none",
        scale=None,
        t_idx=-1,
        b_idx=-1
    ) -> Tensor:
        pre_quant_fp8 = (
            self.quant_alltoall
            and self.algo == 3
            and get_sp_group().ring_world_size == 1
        )
        if not pre_quant_fp8:
            return super().forward(
                attn, query, key, value, seq_lens,
                joint_tensor_query=joint_tensor_query,
                joint_tensor_key=joint_tensor_key,
                joint_tensor_value=joint_tensor_value,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                joint_strategy=joint_strategy,
                scale=scale,
                t_idx=t_idx,
                b_idx=b_idx,
            )
        if self.fa_alltoall_overlap:
            return self._forward_fp8_overlap(query, key, value, seq_lens)
        else:
            return self._forward_fp8_no_overlap(query, key, value, seq_lens)

    def _forward_fp8_no_overlap(self, query, key, value, seq_lens):
        """Non-overlap branch with FP8 pre-quantization before Ulysses All-to-All."""
        origin_dtype = query.dtype
        from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess

        # Rotate Q/K before quantization (same as FP8RotateQuantFA), then quantize in BSND
        q_rot, k_rot = self._get_rot(query.shape[-1], query.device, query.dtype)
        q_fp8, q_scale = fa_block_quant_preprocess(
            torch.matmul(query, q_rot), block_size=128, dst_type=torch_npu.float8_e4m3fn, layout="BSND")
        k_fp8, k_scale = fa_block_quant_preprocess(
            torch.matmul(key, k_rot),   block_size=256, dst_type=torch_npu.float8_e4m3fn, layout="BSND")
        v_fp8, v_scale = fa_block_quant_preprocess(
            value, block_size=256, dst_type=torch_npu.float8_e4m3fn, layout="BSND")
        query = all_to_all_4D(q_fp8, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
        key   = all_to_all_4D(k_fp8, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
        value = all_to_all_4D(v_fp8, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
        # Scale all_to_all: [B, N_all, blocks_local, 1] → [B, N/P, blocks_full, 1]
        q_scale = self._scale_all_to_all(q_scale)
        k_scale = self._scale_all_to_all(k_scale)
        v_scale = self._scale_all_to_all(v_scale)
        s_full = query.shape[1]
        q_scale = q_scale[:, :, :math.ceil(s_full / 128), :]
        k_scale = k_scale[:, :, :math.ceil(s_full / 256), :]
        v_scale = v_scale[:, :, :math.ceil(s_full / 256), :]

        ori_seqlen = query.shape[1]
        if seq_lens is not None and seq_lens < ori_seqlen:
            query_layer, query_pad = query[:, :seq_lens, :, :], query[:, seq_lens:, :, :]
            key_layer, key_pad = key[:, :seq_lens, :, :], key[:, seq_lens:, :, :]
            value_layer, value_pad = value[:, :seq_lens, :, :], value[:, seq_lens:, :, :]
            q_scale = q_scale[:, :, :math.ceil(seq_lens / 128), :]
            k_scale = k_scale[:, :, :math.ceil(seq_lens / 256), :]
            v_scale = v_scale[:, :, :math.ceil(seq_lens / 256), :]
        else:
            query_layer, key_layer, value_layer = query, key, value

        if self.use_all_head:
            out = self._fp8_attn(
                query_layer.transpose(1, 2), key_layer.transpose(1, 2), value_layer.transpose(1, 2),
                origin_dtype, q_scale, k_scale, v_scale,
            )
        else:
            output = []
            for_loop = query_layer.shape[2]
            # algo 3: FA kernel requires contiguous BNSD input.
            # split(1, dim=2) on BSND produces strided views (S-stride = N*D),
            # causing N implicit contiguous copies inside the kernel.
            # One transpose+contiguous upfront gives a contiguous BNSD buffer;
            # splitting along N then yields contiguous per-head chunks — only 3 copies total.
            q_bnsd = query_layer.transpose(1, 2).contiguous()  # [B, N, S, D]
            k_bnsd = key_layer.transpose(1, 2).contiguous()
            v_bnsd = value_layer.transpose(1, 2).contiguous()
            query_layer_list = q_bnsd.split(1, dim=1)          # N × [B, 1, S, D], contiguous
            key_layer_list   = k_bnsd.split(1, dim=1)
            value_layer_list = v_bnsd.split(1, dim=1)
            # _scale_all_to_all ends with transpose → non-contiguous; make contiguous before split
            q_scale_list = q_scale.contiguous().split(1, dim=1)
            k_scale_list = k_scale.contiguous().split(1, dim=1)
            v_scale_list = v_scale.contiguous().split(1, dim=1)
            for i in range(for_loop):
                # query_layer_list[i] is already [B, 1, S, D] BNSD — no transpose needed
                out = self._fp8_attn(
                    query_layer_list[i], key_layer_list[i], value_layer_list[i],
                    origin_dtype, q_scale_list[i], k_scale_list[i], v_scale_list[i],
                )
                output.append(out)
            out = torch.cat(output, dim=2)

        if seq_lens is not None and seq_lens < ori_seqlen:
            # query_pad may be FP8 (pre_quant) or BF16; .to(origin_dtype) is a no-op for BF16
            out_pad = attention_forward(
                query_pad.to(origin_dtype), key_pad.to(origin_dtype), value_pad.to(origin_dtype),
                opt_mode="manual", op_type="fused_attn_score", layout="BSND")
            out = torch.cat([out, out_pad], dim=1)

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = all_to_all_4D(input_=context_layer, scatter_idx=1, gather_idx=2, group=self.ulysses_pg)
        return output

    def _forward_fp8_overlap(self, query, key, value, seq_lens):
        """Overlap branch with FP8 pre-quantization before Ulysses All-to-All."""
        origin_dtype = query.dtype
        from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess

        q_rot, k_rot = self._get_rot(query.shape[-1], query.device, query.dtype)
        q_fp8, q_scale = fa_block_quant_preprocess(
            torch.matmul(query, q_rot), block_size=128, dst_type=torch_npu.float8_e4m3fn, layout="BSND")
        k_fp8, k_scale = fa_block_quant_preprocess(
            torch.matmul(key, k_rot),   block_size=256, dst_type=torch_npu.float8_e4m3fn, layout="BSND")
        v_fp8, v_scale = fa_block_quant_preprocess(
            value, block_size=256, dst_type=torch_npu.float8_e4m3fn, layout="BSND")
        query_layer_list = q_fp8.split(self.ulysess_world_size, dim=2)
        key_layer_list   = k_fp8.split(self.ulysess_world_size, dim=2)
        value_layer_list = v_fp8.split(self.ulysess_world_size, dim=2)
        # Per-chunk scale all_to_all: must match the strided chunk split above.
        # q_fp8 chunk i has heads [i*P:(i+1)*P]; after all_to_all rank r gets head i*P+r.
        # _scale_all_to_all uses a block pattern (rank r gets heads [r*N_local:(r+1)*N_local))
        # which MISMATCHES the strided chunk ordering — so we use _chunk_scale_a2a instead.
        s_full = query.shape[1] * self.ulysess_world_size
        n_chunks = len(query_layer_list)
        q_scale_list = [self._chunk_scale_a2a(q_scale, i, 128, s_full) for i in range(n_chunks)]
        k_scale_list = [self._chunk_scale_a2a(k_scale, i, 256, s_full) for i in range(n_chunks)]
        v_scale_list = [self._chunk_scale_a2a(v_scale, i, 256, s_full) for i in range(n_chunks)]

        for_loop = len(query_layer_list)

        output_fa = []
        qkv_event = torch.npu.Event()
        qkv_event.record()
        q_lists, k_lists, v_lists = [], [], []

        with torch.npu.stream(self.current_stream):
            query_layer = all_to_all_4D(input_=query_layer_list[0], scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
            key_layer = all_to_all_4D(input_=key_layer_list[0], scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
            value_layer = all_to_all_4D(input_=value_layer_list[0], scatter_idx=2, gather_idx=1, group=self.ulysses_pg)

            ori_seqlen = query_layer.shape[1]
            need_pad = False

            if seq_lens is not None and seq_lens < ori_seqlen:
                need_pad = True
                # 照搬 else 分支逻辑：切分出 pad 部分，并将 query_layer 重赋值为有效部分
                query_layer, query_pad = query_layer[:, :seq_lens, :, :], query_layer[:, seq_lens:, :, :]
                key_layer, key_pad = key_layer[:, :seq_lens, :, :], key_layer[:, seq_lens:, :, :]
                value_layer, value_pad = value_layer[:, :seq_lens, :, :], value_layer[:, seq_lens:, :, :]

            q_lists.append(query_layer)
            k_lists.append(key_layer)
            v_lists.append(value_layer)

            qs = q_scale_list[0][:, :, :math.ceil(q_lists[0].shape[1] / 128), :]
            ks = k_scale_list[0][:, :, :math.ceil(k_lists[0].shape[1] / 256), :]
            vs = v_scale_list[0][:, :, :math.ceil(v_lists[0].shape[1] / 256), :]
            out = self._fp8_attn(
                q_lists[0].transpose(1, 2), k_lists[0].transpose(1, 2), v_lists[0].transpose(1, 2),
                origin_dtype, qs, ks, vs,
            )

            if need_pad:
                out_pad = attention_forward(
                    query_pad.to(origin_dtype), key_pad.to(origin_dtype), value_pad.to(origin_dtype),
                    opt_mode="manual", op_type="fused_attn_score", layout="BSND")
                out = torch.cat([out, out_pad], dim=1)

            output_fa.append(out)
            self.event[0].record()

        with torch.npu.stream(self.stream2):
            for i in range(1, for_loop):
                self.stream2.wait_event(qkv_event)
                # B, S, 1, D
                query_layer = all_to_all_4D(input_=query_layer_list[i], scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
                key_layer = all_to_all_4D(input_=key_layer_list[i], scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
                value_layer = all_to_all_4D(input_=value_layer_list[i], scatter_idx=2, gather_idx=1, group=self.ulysses_pg)

                self.event[i].record(self.stream2)
                q_lists.append(query_layer)
                k_lists.append(key_layer)
                v_lists.append(value_layer)

        for i in range(1, for_loop):
            self.current_stream.wait_event(self.event[i])
            query_layer = q_lists[i]
            key_layer = k_lists[i]
            value_layer = v_lists[i]

            ori_seqlen = query_layer.shape[1]
            need_pad = False
            if seq_lens is not None and seq_lens < ori_seqlen:
                need_pad = True
                # 照搬 else 分支逻辑：切分出 pad 部分，并将 query_layer 重赋值为有效部分
                query_layer, query_pad = query_layer[:, :seq_lens, :, :], query_layer[:, seq_lens:, :, :]
                key_layer, key_pad = key_layer[:, :seq_lens, :, :], key_layer[:, seq_lens:, :, :]
                value_layer, value_pad = value_layer[:, :seq_lens, :, :], value_layer[:, seq_lens:, :, :]

            qs = q_scale_list[i][:, :, :math.ceil(query_layer.shape[1] / 128), :]
            ks = k_scale_list[i][:, :, :math.ceil(key_layer.shape[1] / 256), :]
            vs = v_scale_list[i][:, :, :math.ceil(value_layer.shape[1] / 256), :]
            out = self._fp8_attn(
                query_layer.transpose(1, 2), key_layer.transpose(1, 2), value_layer.transpose(1, 2),
                origin_dtype, qs, ks, vs,
            )

            if need_pad:
                out_pad = attention_forward(
                    query_pad.to(origin_dtype), key_pad.to(origin_dtype), value_pad.to(origin_dtype),
                    opt_mode="manual", op_type="fused_attn_score", layout="BSND")
                out = torch.cat([out, out_pad], dim=1)

            output_fa.append(out)
            self.event[i].record()

        output_res = []
        with torch.npu.stream(self.stream2):
            for i in range(for_loop):
                self.stream2.wait_event(self.event[i])
                # BSND gather 2 scatetrer 1
                output_tmp = all_to_all_4D(input_=output_fa[i], scatter_idx=1, gather_idx=2, group=self.ulysses_pg)
                self.event[i].record(self.stream2)
                output_res.append(output_tmp)
        for i in range(for_loop):
            self.current_stream.wait_event(self.event[i])
        output = torch.cat(output_res, dim=2)
        return output
