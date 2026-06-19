import struct
from pathlib import Path
from typing import List

import numpy as np
import pytest

from olmo_core.data import (
    MegatronFSLDataset,
    MegatronFSLDatasetConfig,
    MegatronIdx,
    TokenizerConfig,
)
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.exceptions import OLMoConfigurationError

# A dolma2-like setup: EOS is well above the uint16 range, so the int32 tokens MUST be read
# back as uint32 (not uint16) to round-trip correctly.
EOS = 100257
VOCAB_SIZE = 100278

# Megatron dtype code 4 == int32.
_INT32_CODE = 4


def _write_megatron(base: Path, docs: List[List[int]], dtype_code: int = _INT32_CODE) -> int:
    """Write a tiny Megatron ``.bin`` + ``.idx`` pair. Returns the total token count."""
    token_dtype = MegatronIdx._CODE_TO_DTYPE[dtype_code]
    item_size = np.dtype(token_dtype).itemsize

    flat = np.concatenate([np.array(d, dtype=token_dtype) for d in docs])
    flat.tofile(str(base) + ".bin")

    lengths = np.array([len(d) for d in docs], dtype=np.int32)
    pointers = np.zeros(len(docs), dtype=np.int64)
    acc = 0
    for i, length in enumerate(lengths):
        pointers[i] = acc
        acc += int(length) * item_size
    doc_indices = np.arange(len(docs) + 1, dtype=np.int64)

    with open(str(base) + ".idx", "wb") as f:
        f.write(MegatronIdx.MAGIC)
        f.write(struct.pack("<Q", 1))  # version
        f.write(struct.pack("<B", dtype_code))
        f.write(struct.pack("<Q", len(docs)))  # sequence_count
        f.write(struct.pack("<Q", len(doc_indices)))  # doc_index_count
        f.write(lengths.tobytes())
        f.write(pointers.tobytes())
        f.write(doc_indices.tobytes())

    return int(lengths.sum())


def _docs() -> List[List[int]]:
    # Each "document" ends with the EOS id, mimicking Megatron --append-eod.
    return [
        [5, 9, 70000, 3, EOS],  # includes a token > uint16 max
        [1, 2, 3, EOS],
        [42, 100256, 7, 8, 9, EOS],
    ]


def test_megatron_idx_parse(tmp_path: Path):
    base = tmp_path / "merged_text"
    total = _write_megatron(base, _docs())

    idx = MegatronIdx(base.with_suffix(".idx"))
    assert idx.token_dtype is np.int32
    assert idx.item_size == 4
    assert idx.sequence_count == 3
    assert idx.doc_index_count == 4
    assert idx.total_tokens == total

    # Validation passes against the matching .bin.
    idx.validate_against_bin(str(base) + ".bin")


def test_megatron_idx_rejects_non_megatron(tmp_path: Path):
    bad = tmp_path / "bad.idx"
    bad.write_bytes(b"NOTMEGATRON")
    with pytest.raises(OLMoConfigurationError):
        MegatronIdx(bad)


def test_megatron_fsl_dataset_reads_tokens(tmp_path: Path):
    docs = _docs()
    base = tmp_path / "merged_text"
    total = _write_megatron(base, docs)
    flat = np.concatenate([np.array(d, dtype=np.int32) for d in docs])

    seq_len = 4
    ds = MegatronFSLDataset(
        str(base) + ".bin",
        sequence_length=seq_len,
        pad_token_id=EOS,
        eos_token_id=EOS,
        vocab_size=VOCAB_SIZE,
        dtype=np.uint32,
    )
    ds.prepare()

    assert ds.num_tokens == total  # exact, from the .idx (not floored)
    assert len(ds) == total // seq_len

    # The int32 bytes are reinterpreted as uint32 bit-for-bit (valid for non-negative ids),
    # including the token > uint16 max.
    expected = flat.astype(np.uint32)
    got = np.concatenate([ds[i]["input_ids"].numpy() for i in range(len(ds))])
    assert got.tolist() == expected[: len(ds) * seq_len].tolist()
    assert 70000 in got.tolist()


def test_megatron_fsl_dataset_dtype_mismatch(tmp_path: Path):
    base = tmp_path / "merged_text"
    _write_megatron(base, _docs())
    # int32 file (item size 4) read as uint16 (item size 2) must be rejected.
    with pytest.raises(OLMoConfigurationError):
        MegatronFSLDataset(
            str(base) + ".bin",
            sequence_length=4,
            pad_token_id=EOS,
            eos_token_id=EOS,
            vocab_size=VOCAB_SIZE,
            dtype=np.uint16,
        )


def test_megatron_fsl_dataset_truncated_bin_fails(tmp_path: Path):
    base = tmp_path / "merged_text"
    _write_megatron(base, _docs())

    # Lop off the last few tokens of the .bin to simulate an incomplete / partial merge.
    bin_path = Path(str(base) + ".bin")
    with open(bin_path, "r+b") as f:
        f.truncate(bin_path.stat().st_size - 4 * 3)

    ds = MegatronFSLDataset(
        str(base) + ".bin",
        sequence_length=4,
        pad_token_id=EOS,
        eos_token_id=EOS,
        vocab_size=VOCAB_SIZE,
        dtype=np.uint32,
    )
    with pytest.raises(OLMoConfigurationError):
        ds.prepare()


def test_megatron_fsl_dataset_config_build(tmp_path: Path):
    base = tmp_path / "merged_text"
    total = _write_megatron(base, _docs())

    config = MegatronFSLDatasetConfig(
        paths=[str(base) + ".bin"],
        sequence_length=4,
        tokenizer=TokenizerConfig.dolma2(),
        dtype=NumpyDatasetDType.uint32,
        work_dir=str(tmp_path / "work"),
    )
    ds = config.build()
    ds.prepare()
    assert isinstance(ds, MegatronFSLDataset)
    assert ds.num_tokens == total
    assert ds.eos_token_id == EOS

    # dolma2 vocab (100278) auto-selects uint32 even without an explicit dtype.
    config_auto = MegatronFSLDatasetConfig(
        paths=[str(base) + ".bin"],
        sequence_length=4,
        tokenizer=TokenizerConfig.dolma2(),
    )
    assert config_auto.get_dtype() is np.uint32
