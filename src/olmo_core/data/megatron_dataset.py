"""
Zero-copy support for `Megatron-LM <https://github.com/NVIDIA/Megatron-LM>`_
``MMapIndexedDataset`` token files (a raw ``.bin`` of token IDs plus a sidecar ``.idx``
holding per-document lengths and offsets).

The token ``.bin`` is read *in place* — no conversion or copy — by reusing the raw,
header-agnostic read path of :class:`~olmo_core.data.numpy_dataset.NumpyFSLDataset`.
Megatron stores token IDs as a *signed* integer type (commonly ``int32``), but token IDs
are always non-negative and well below ``2**31``, so reinterpreting the same bytes as the
matching *unsigned* type (``uint32``) is a free, bit-exact operation. The sidecar ``.idx``
is used only off the hot path: for exact token/document accounting and to validate that the
``.bin`` is complete (it is never consulted to read tokens for plain fixed-sequence-length
chunking).

.. seealso::
    :class:`~olmo_core.data.numpy_dataset.NumpyFSLDataset` for the underlying chunking
    behavior (token IDs from all files are concatenated and split into contiguous
    ``sequence_length`` blocks; EOS separators are expected to already be present in the
    data, which is the case for Megatron files written with ``--append-eod``).
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ..aliases import PathOrStr
from ..exceptions import OLMoConfigurationError
from ..io import get_file_size
from .numpy_dataset import NumpyDatasetBase, NumpyDatasetConfig, NumpyFSLDataset

__all__ = ["MegatronIdx", "MegatronFSLDataset", "MegatronFSLDatasetConfig", "megatron_idx_path"]

log = logging.getLogger(__name__)


def megatron_idx_path(bin_path: PathOrStr) -> Path:
    """
    Get the path to the sidecar ``.idx`` file for a Megatron ``.bin`` token file.

    :param bin_path: Path to the ``.bin`` token file.

    :returns: The corresponding ``.idx`` path (``foo.bin`` -> ``foo.idx``).
    """
    return Path(str(bin_path)).with_suffix(".idx")


class MegatronIdx:
    """
    Parser for a Megatron ``MMapIndexedDataset`` sidecar ``.idx`` file.

    Only the header is read eagerly; the (potentially large) per-sequence length array is
    read lazily and cached when :data:`total_tokens` is first accessed.

    The on-disk layout is: a 9-byte magic ``b"MMIDIDX\\x00\\x00"``, a ``uint64`` version, a
    one-byte dtype code, a ``uint64`` sequence count, a ``uint64`` document-index count, then
    an ``int32`` array of per-sequence lengths, an ``int64`` array of byte pointers, and an
    ``int64`` document index.

    :param idx_path: Path to the ``.idx`` file.

    :raises OLMoConfigurationError: If the file is not a recognized Megatron index or uses an
        unsupported token dtype.
    """

    MAGIC = b"MMIDIDX\x00\x00"
    _HEADER_SIZE = 9 + 8 + 1 + 8 + 8

    # Megatron dtype codes -> numpy dtype. Token-ID files are integer types; the others are
    # included for completeness so we can give a clear error rather than a silent misread.
    _CODE_TO_DTYPE: Dict[int, Any] = {
        1: np.uint8,
        2: np.int8,
        3: np.int16,
        4: np.int32,
        5: np.int64,
        6: np.float32,
        7: np.float64,
        8: np.uint16,
    }

    def __init__(self, idx_path: PathOrStr):
        self.path = Path(str(idx_path))
        if not self.path.is_file():
            raise OLMoConfigurationError(f"Megatron index file not found: '{self.path}'")
        with self.path.open("rb") as f:
            if f.read(9) != self.MAGIC:
                raise OLMoConfigurationError(
                    f"'{self.path}' is not a Megatron MMapIndexedDataset (.idx) file"
                )
            (self.version,) = struct.unpack("<Q", f.read(8))
            (dtype_code,) = struct.unpack("<B", f.read(1))
            (self.sequence_count,) = struct.unpack("<Q", f.read(8))
            (self.doc_index_count,) = struct.unpack("<Q", f.read(8))

        if dtype_code not in self._CODE_TO_DTYPE:
            raise OLMoConfigurationError(
                f"unsupported Megatron token dtype code {dtype_code} in '{self.path}'"
            )
        self.token_dtype = self._CODE_TO_DTYPE[dtype_code]
        self.item_size: int = np.dtype(self.token_dtype).itemsize

    @cached_property
    def total_tokens(self) -> int:
        """
        The exact total number of tokens, summed from the per-sequence lengths in the index.
        """
        lengths = np.memmap(
            self.path,
            dtype=np.int32,
            mode="r",
            offset=self._HEADER_SIZE,
            shape=(self.sequence_count,),
        )
        return int(lengths.sum(dtype=np.int64))

    def validate_against_bin(self, bin_path: PathOrStr) -> None:
        """
        Check that the ``.bin`` token file is consistent with this index, i.e. that it
        contains exactly ``total_tokens`` tokens. This catches a truncated or partially-merged
        ``.bin`` before any training time is spent.

        :param bin_path: Path to the ``.bin`` token file.

        :raises OLMoConfigurationError: If the ``.bin`` size does not match the index.
        """
        expected = self.total_tokens * self.item_size
        actual = get_file_size(bin_path)
        if actual != expected:
            raise OLMoConfigurationError(
                f"Megatron token file '{bin_path}' is {actual:,} bytes but its index "
                f"'{self.path}' implies {expected:,} bytes "
                f"({self.total_tokens:,} tokens x {self.item_size}B). "
                "The .bin is likely incomplete or does not match its .idx."
            )


class MegatronFSLDataset(NumpyFSLDataset):
    """
    A fixed-sequence-length dataset backed by raw Megatron ``.bin`` token files, read in place
    (zero-copy). Token IDs from all files are concatenated and chunked into contiguous
    ``sequence_length`` blocks, exactly like :class:`~olmo_core.data.numpy_dataset.NumpyFSLDataset`.

    The only behavioral differences from the base class are:

    - Token and instance counts are sourced from the sidecar ``.idx`` rather than from the
      ``.bin`` file size, and :data:`num_tokens` reports the *exact* token total.
    - :meth:`prepare` validates each ``.bin`` against its ``.idx`` so an incomplete file fails
      loudly up front.

    The ``dtype`` must match the on-disk token width (e.g. ``uint32`` for an ``int32`` Megatron
    file); a mismatch is rejected at construction time.

    :param paths: Paths to Megatron ``.bin`` token files.
    :param dtype: The (unsigned) numpy dtype used to read the tokens. Must have the same item
        size as the Megatron file's token dtype.
    """

    def __init__(self, *paths: PathOrStr, **kwargs: Any):
        super().__init__(*paths, **kwargs)
        item_size = self.dtype(0).itemsize
        self._idx: Dict[str, MegatronIdx] = {}
        for path in self.paths:
            idx = MegatronIdx(megatron_idx_path(path))
            if idx.item_size != item_size:
                raise OLMoConfigurationError(
                    f"configured dtype '{np.dtype(self.dtype).name}' (item size {item_size}) "
                    f"does not match the Megatron token width (item size {idx.item_size}) "
                    f"for '{path}'. Set the dataset dtype to an unsigned type of the same width."
                )
            self._idx[str(path)] = idx

    def _get_file_size_and_length(self, path: PathOrStr, idx: int, dtype=None) -> Tuple[int, int]:
        # Source the token count from the .idx (authoritative) instead of os.stat. For label
        # mask files (not part of a Megatron dataset) fall back to the base implementation.
        key = str(path)
        if key not in self._idx:
            return super()._get_file_size_and_length(path, idx, dtype=dtype)
        del idx
        item_size = (dtype or self.dtype)(0).itemsize
        num_tokens = self._idx[key].total_tokens
        return num_tokens * item_size, num_tokens // self.sequence_length

    @property
    def num_tokens(self) -> int:
        return sum(idx.total_tokens for idx in self._idx.values())

    def prepare(self):
        for path in self.paths:
            self._idx[str(path)].validate_against_bin(path)
        super().prepare()
        if self.fs_local_rank == 0:
            num_instances = len(self)
            used = num_instances * self.sequence_length
            num_seqs = sum(idx.sequence_count for idx in self._idx.values())
            log.info(
                f"Megatron FSL dataset ready: {len(self.paths)} file(s), "
                f"{self.num_tokens:,} tokens in {num_seqs:,} sequences -> "
                f"{num_instances:,} instances of length {self.sequence_length} "
                f"({self.num_tokens - used:,} trailing tokens dropped)."
            )


@dataclass(kw_only=True)
class MegatronFSLDatasetConfig(NumpyDatasetConfig):
    """
    Configuration for a :class:`MegatronFSLDataset`.

    Set :data:`~olmo_core.data.numpy_dataset.NumpyDatasetConfig.paths` to one or more Megatron
    ``.bin`` token files (each must have a sibling ``.idx``). For an ``int32`` Megatron file set
    :data:`~olmo_core.data.numpy_dataset.NumpyDatasetConfig.dtype` to ``"uint32"`` (or leave it
    unset to let it be inferred from the tokenizer's vocab size).
    """

    sequence_length: int
    """
    The number of tokens per instance. Generally this should match your model's max input length.
    """

    def validate(self):
        if self.sequence_length <= 0:
            raise OLMoConfigurationError("'sequence_length' must be positive")

    def build(self) -> NumpyDatasetBase:
        self.validate()
        paths, metadata, _ = self._resolve_paths_metadata(allow_mix=False)
        dataset = MegatronFSLDataset(
            *paths,
            sequence_length=self.sequence_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            vocab_size=self.tokenizer.vocab_size,
            dtype=self.get_dtype(),
            metadata=metadata,
            include_instance_metadata=self.include_instance_metadata,
            bos_token_id=self.tokenizer.bos_token_id,
            instance_filter_config=self.instance_filter_config,
        )
        return self._finalize(dataset)
