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
            attn_type = attn_type,
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
                            num_heads = 1, input_layout = "BNSD", scale = scale, pre_tokens=2147483647, next_tokens=2147483647)[0]
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
                out_pad = attention_forward(query_pad, key_pad, value_pad,
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