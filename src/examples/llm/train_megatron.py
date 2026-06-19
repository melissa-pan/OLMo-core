"""
Train a transformer language model on pre-tokenized **Megatron-LM** data
(a raw ``.bin`` of token IDs + a sidecar ``.idx``), read zero-copy via
:class:`~olmo_core.data.MegatronFSLDatasetConfig`.

This is a Megatron-data variant of ``train.py`` in this directory — diff the two to see
exactly what changes when you swap the data source. Launch with torchrun:

    torchrun --nproc-per-node=4 src/examples/llm/train_megatron.py run_name [OVERRIDES...]

The defaults below point at a colleague's dolma corpus tokenized with the OLMo-3 / dolma2
tokenizer. Adjust ``DATA_PATHS``, the model factory, and ``--sequence-length`` for your run.
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
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    ProfilerCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

# Paths to the Megatron ``.bin`` token file(s). Each must have a sibling ``.idx``.
# This is a list so you can add more/different Megatron files later — they are concatenated.
DATA_PATHS = [
    "/work/mayank/dolma-fixed/merged_text.bin",
]


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

    # Build components (identical wiring to train.py — only the dataset config differs).
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    if not trainer.no_checkpoints:
        trainer.maybe_load_checkpoint()

    trainer.fit()


def build_config(opts, overrides: List[str]) -> ExperimentConfig:
    save_folder = opts.save_folder or f"/tmp/{opts.run_name}"
    work_dir = opts.work_dir or "/tmp/dataset-cache"

    # The Megatron data was tokenized with the OLMo-3 / dolma2 tokenizer (vocab 100,278,
    # EOS 100,257). This MUST match the tokenizer that produced the .bin.
    tokenizer_config = TokenizerConfig.dolma2()

    try:
        factory = getattr(TransformerConfig, opts.model_factory)
    except AttributeError:
        raise ValueError(f"Unknown model factory: {opts.model_factory}")
    # The model's vocab (embedding + LM head) is sized to the tokenizer, padded to a multiple
    # of 128 for throughput. This is what makes token IDs from the data valid lookups.
    model_config = factory(vocab_size=tokenizer_config.padded_vocab_size())

    # Zero-copy Megatron loader: the int32 .bin is read in place as uint32, and the .idx is
    # used to validate completeness + report exact token counts.
    dataset_config = MegatronFSLDatasetConfig(
        paths=DATA_PATHS,
        sequence_length=opts.sequence_length,
        tokenizer=tokenizer_config,
        dtype="uint32",  # int32 Megatron tokens read as uint32 (free, bit-exact)
        work_dir=work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=256 * 1024,  # in TOKENS, not instances
        seed=0,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=16 * 1024,  # in TOKENS
        max_sequence_length=opts.sequence_length,
        optim=AdamWConfig(
            lr=1e-3,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=100),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(save_interval=1000, ephemeral_save_interval=100, save_async=True),
        )
        .with_callback(
            "wandb",
            WandBCallback(name=opts.run_name, cancel_check_interval=10, enabled=False),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
    )
    # NOTE on evaluation: this script intentionally has NO evaluators.
    #   * For a first run, watch the *training loss* curve fall — that alone confirms the
    #     data -> model -> optimizer pipeline is wired correctly.
    #   * For a real health signal, add an LMEvaluatorCallbackConfig pointed at a held-out
    #     validation set tokenized with the SAME (dolma2) tokenizer — e.g. a small held-out
    #     Megatron .bin via MegatronFSLDatasetConfig. Do NOT reuse train.py's C4/gpt2 eval set:
    #     it is gpt2-tokenized and incompatible with this 100,278-vocab model.
    #   * Downstream benchmarks (e.g. DownstreamEvaluatorCallbackConfig tasks=["hellaswag"]) are
    #     now tokenizer-consistent with dolma2 and can be added once the run is stable.

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
        description="Train a transformer LM on Megatron-format pre-tokenized data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_name", type=str, help="The name of the run.")
    parser.add_argument(
        "--model-factory",
        type=str,
        default="llama2_271M",
        help="Any classmethod on TransformerConfig (e.g. llama2_271M).",
    )
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--save-folder", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the config and exit.")
    opts, overrides = parser.parse_known_args()
    return opts, overrides


def main():
    opts, overrides = parser_args()
    config = build_config(opts, overrides)

    if opts.dry_run:
        rich.print(config)
        return

    prepare_training_environment()
    try:
        train(config)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
