"""
Throughput-profiling experiment: an OLMo-core port of a colleague's dolomite/Megatron-LM
MoE pre-training setup (``training_setup.yaml``), reading the colleague's pre-tokenized
dolma corpus *zero-copy* via :class:`~olmo_core.data.MegatronFSLDatasetConfig`.

Goal: run for ~3 hours and measure steady-state training throughput ("billion tokens/day").
The learning-rate schedule and the full 250k-step budget are irrelevant to throughput, so we
do NOT try to finish them — we just run long enough to read a stable tokens/sec from the
built-in :class:`~olmo_core.train.callbacks.SpeedMonitorCallback`.

Launch on the local 8x B200 box with torchrun, bounding wall-clock with ``timeout``::

    timeout 3h torchrun --nproc-per-node=8 src/examples/llm/profile_dolma_moe.py train-billion_tokens_per_day

Inspect the adaptation without a GPU (prints the model size + batch math and exits)::

    python src/examples/llm/profile_dolma_moe.py train-billion_tokens_per_day --dry-run

--------------------------------------------------------------------------------------------
Mapping from the colleague's ``training_setup.yaml`` (dolomite ``gpt_base`` MoE) to OLMo-core
--------------------------------------------------------------------------------------------
  hidden_size 1536, num_layers 40                  -> d_model=1536, n_layers=40
  24 attention + 24 KV heads (full MHA), head 64   -> n_heads=24, n_kv_heads=24, head_dim=64
  rmsnorm, rope, no biases, swiglu                 -> llama_like defaults (RMSNorm/RoPE/SwiGLU, bias=False)
  MoE: 128 experts, top-8, expert ffn 256          -> num_experts=128, top_k=8, expert_hidden_size=256
  shared expert ffn 1024                           -> shared_expert_hidden_size=1024
  vocab 100278, eos/bos 100257, pad 100277         -> TokenizerConfig.dolma2() (padded vocab)
  AdamW lr 3e-4 wd 0.1 betas (.9,.95) eps 1e-10    -> AdamWConfig(...)  (+ no weight decay on embeddings)
  cosine, 2500 warmup                              -> CosWithWarmup(warmup=2500)
  bf16, ZeRO stage 3, torch_compile, dp shard 8    -> FSDP (param bf16 / reduce fp32), compile_model=True
  micro 6 x grad-accum 1 x dp 8 @ seqlen 4096      -> global_batch 196608 tok, rank_microbatch 24576 tok

Deviations (all throughput-neutral, noted for honesty):
  * vocab is padded to a multiple of 128 (OLMo convention; +~0.1% embedding params).
  * MoE uses the default capacity-based experts (plain torch.bmm) rather than a dropless
    grouped-GEMM kernel, so no GPU kernel needs to be compiled. For a production-faithful
    (dropless) number, install `grouped_gemm` and pass --dropless.
  * attention uses whatever backend is available (flash-attn if installed, else torch SDPA).
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import List, cast

import rich

from olmo_core.config import Config, DType
from olmo_core.data import (
    MegatronFSLDatasetConfig,
    NumpyDataLoaderConfig,
    TokenizerConfig,
)
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

# The colleague's pre-tokenized dolma corpus (157.5B tokens, dolma2/OLMo-3 tokenizer).
# Each ``.bin`` must have a sibling ``.idx``; they are read in place (zero-copy).
DATA_PATHS = [
    "/work/mayank/dolma-fixed/merged_text.bin",
]

# ---- architecture, straight from training_setup.yaml (model_args.pretrained_config) ----
D_MODEL = 1536
N_LAYERS = 40
N_HEADS = 24
N_KV_HEADS = 24  # == n_heads -> full multi-head attention (not GQA)
HEAD_DIM = 64  # 1536 / 24
NUM_EXPERTS = 128
TOP_K = 8  # num_experts_per_tok
EXPERT_HIDDEN_SIZE = 256  # per-routed-expert intermediate_size
SHARED_EXPERT_HIDDEN_SIZE = 1024  # shared_intermediate_size (always-active shared expert)
ROPE_THETA = 500_000  # OLMo-3 convention; throughput-neutral
LAYER_NORM_EPS = 1e-5  # matches yaml layer_norm_epsilon


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    train_module: TransformerTrainModuleConfig
    init_seed: int = 12536


def train(config: ExperimentConfig):
    if get_rank() == 0:
        rich.print(config)

    seed_all(config.init_seed)

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Pure throughput profile: no checkpoints to load.
    trainer.fit()


def build_config(opts, overrides: List[str]) -> ExperimentConfig:
    save_folder = opts.save_folder or f"/tmp/{opts.run_name}"
    work_dir = opts.work_dir or "/tmp/dolma-moe-cache"

    # Tokenizer that produced the .bin (dolma2 / OLMo-3): vocab 100278, eos 100257, pad 100277.
    tokenizer_config = TokenizerConfig.dolma2()

    # ~6.9B-total / ~1.0B-active fine-grained MoE with a shared expert.
    model_config = TransformerConfig.llama_like_moe(
        d_model=D_MODEL,
        vocab_size=tokenizer_config.padded_vocab_size(),
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        expert_hidden_size=EXPERT_HIDDEN_SIZE,
        shared_expert_hidden_size=SHARED_EXPERT_HIDDEN_SIZE,
        dropless=opts.dropless,
        reordered_norm=False,  # standard pre-norm, like dolomite gpt_base
        qk_norm=False,  # no QK-norm in the yaml
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
        rope_theta=ROPE_THETA,
        layer_norm_eps=LAYER_NORM_EPS,
    )

    # Zero-copy Megatron loader: int32 .bin read in place as uint32; .idx used for validation.
    dataset_config = MegatronFSLDatasetConfig(
        paths=opts.data_path,
        sequence_length=opts.sequence_length,
        tokenizer=tokenizer_config,
        dtype="uint32",
        work_dir=work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=opts.global_batch_size,  # in TOKENS
        seed=0,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.rank_microbatch_size,  # in TOKENS
        max_sequence_length=opts.sequence_length,
        optim=AdamWConfig(
            lr=3e-4,
            betas=(0.9, 0.95),
            eps=1e-10,
            weight_decay=0.1,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,  # ZeRO-3 (full shard) across all data-parallel GPUs
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup=2500),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            # Mirror the colleague's 250k-step budget; the run is actually bounded to ~3h by
            # `timeout` at launch (or --max-steps for a smoke test). Throughput stabilizes in
            # minutes, well before either bound.
            max_duration=Duration.steps(opts.max_steps),
            metrics_collect_interval=5,
            cancel_check_interval=5,
            no_checkpoints=True,  # pure throughput profile
            no_evals=True,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "wandb",
            WandBCallback(name=opts.run_name, cancel_check_interval=10, enabled=False),
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )
    # SpeedMonitorCallback is auto-added by the Trainer; it logs `throughput/device/TPS`
    # (+ "(actual avg)") to the console. Aggregate throughput = per-device TPS x num GPUs,
    # which equals global_batch_size x steps/sec. tokens/day = that x 86400.

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    )
    return config.merge(overrides)


def parser_args():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"python {sys.argv[0]} RUN_NAME [OPTIONS...] [CONFIG_OVERRIDES...]",
        description="Profile training throughput for the dolomite-MoE setup ported to OLMo-core.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_name", type=str, nargs="?", default="train-billion_tokens_per_day")
    parser.add_argument(
        "--data-path",
        type=str,
        nargs="+",
        default=DATA_PATHS,
        help="Megatron .bin file(s) to train on (each needs a sibling .idx). "
        "Override for other hosts where the corpus is mounted elsewhere.",
    )
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument(
        "--global-batch-size", type=int, default=48 * 4096, help="Global batch size in TOKENS."
    )
    parser.add_argument(
        "--rank-microbatch-size",
        type=int,
        default=6 * 4096,
        help="Per-GPU micro-batch size in TOKENS.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=250_000,
        help="Step budget (the run is wall-clock bounded by `timeout` at launch).",
    )
    parser.add_argument(
        "--dropless",
        action="store_true",
        help="Use dropless MoE (needs the grouped_gemm kernel for speed).",
    )
    parser.add_argument("--save-folder", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the config + sizes and exit.")
    opts, overrides = parser.parse_known_args()
    return opts, overrides


def _print_summary(config: ExperimentConfig, opts):
    mc = config.model
    seq = opts.sequence_length
    gbs, mbs = opts.global_batch_size, opts.rank_microbatch_size
    rich.print(config)
    print("\n================ experiment summary ================")
    print(
        f"model: d_model={D_MODEL} n_layers={N_LAYERS} experts={NUM_EXPERTS} top_k={TOP_K} "
        f"(+shared) seqlen={seq}"
    )
    print(f"  total params        : {mc.num_params:,}")
    print(f"  active params/token : {mc.num_active_params:,}")
    print(f"  non-embedding params: {mc.num_non_embedding_params:,}")
    print(
        f"batch: global={gbs:,} tok ({gbs // seq} seqs/step) | "
        f"rank micro={mbs:,} tok ({mbs // seq} seqs)"
    )
    print(f"  -> on 8 GPUs: {gbs // 8:,} tok/GPU/step = {gbs // 8 // mbs} grad-accum step(s)")
    print("throughput readout: watch console `throughput/device/TPS (actual avg)`")
    print("  tokens/day ~= TPS_per_device x 8 x 86400  (== global_batch x steps/sec x 86400)")
    print("====================================================\n")


def main():
    opts, overrides = parser_args()
    config = build_config(opts, overrides)

    if opts.dry_run:
        _print_summary(config, opts)
        return

    prepare_training_environment(seed=12536)
    try:
        train(config)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
