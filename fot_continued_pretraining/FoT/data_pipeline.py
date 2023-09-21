import logging
import random
import sys
import time
from collections import defaultdict
from functools import partial
from typing import Iterable, List, Tuple, Dict, Generator, Callable

import numpy as np
from absl import flags

LOGGER = logging.Logger("DataPipeline", level=logging.INFO)
LOGGER_HANDLER = logging.StreamHandler(sys.stderr)
LOGGER_HANDLER.setFormatter(logging.Formatter("[%(asctime)s] FoT Tunning [%(levelname)s] : %(message)s"))
LOGGER.addHandler(LOGGER_HANDLER)



class DataPipeline:
    """
    Base class for the data pipeline.
    token_source should generate tuples consisting of
    (tokens_from_doc, loss_mask, name_of_the_data_source)
    Tokens should be generated on a per doc/example basis.
    That is, each generated tuple should contain all tokens
    from the document/example.
    """

    def __init__(
        self, token_source: Iterable[Tuple[List[int], List[float], str]], batch_size: int, seq_len: int
    ) -> None:
        self.token_source = token_source
        self.batch_size = batch_size
        self.seq_len = seq_len

    def __iter__(self):
        raise NotImplementedError()


class LinearPipeline(DataPipeline):
    """
    Pipeline that concatenates docs/examples sequentially to create the batch.
    """

    def __init__(
        self, token_source: Iterable[Tuple[List[int], List[float], str]], batch_size: int, seq_len: int
    ) -> None:
        super().__init__(token_source, batch_size, seq_len)

    def __iter__(self):
        token_src = iter(self.token_source)
        total_next = 0
        total_tokens = 0
        token_buffer = []
        loss_mask_buffer = []
        chunk_size = self.batch_size * self.seq_len
        while True:
            # as we predict next token we need chunk_size + 1
            while len(token_buffer) < chunk_size + 1:
                tokens, loss_mask, _ = next(token_src)
                total_next += 1
                token_buffer.extend(tokens)
                loss_mask_buffer.extend(loss_mask)

            assert len(token_buffer) == len(loss_mask_buffer)

            data_batch = {
                "input_tokens": np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                    self.batch_size, self.seq_len
                ),
                "target_tokens": np.array(token_buffer[1 : chunk_size + 1], dtype=np.int32).reshape(
                    self.batch_size, self.seq_len
                ),
                "loss_masks": np.array(loss_mask_buffer[1 : chunk_size + 1], dtype=np.float32).reshape(
                    self.batch_size, self.seq_len
                ),
            }

            total_tokens += chunk_size
            yield data_batch, {
                "dataset_example_index": total_next,
                "dataset_total_tokens": total_tokens,
            }

            token_buffer = token_buffer[chunk_size:]
            loss_mask_buffer = loss_mask_buffer[chunk_size:]


class DocAwareDataPipeline(DataPipeline):
    """
    Pipeline that assigns documents to the indexes of the batch.
    To be more precise, each document from token_source is assigned an index in the batch.
    The place is populated by the document's tokens till the end.
    After that, a new document is assigned to this place in the batch.
    """

    def __init__(
        self,
        token_source: Iterable[Tuple[List[int], List[float], str]],
        batch_size: int,
        seq_len: int,
        pad: bool = False,
    ) -> None:
        super().__init__(token_source=token_source, batch_size=batch_size, seq_len=seq_len)
        self.pad = pad

    def __iter__(self):
        token_src = iter(self.token_source)
        doc_lm_list = [[[], [], []] for _ in range(self.batch_size)]
        total_next = 0
        total_tokens = 0

        def populate_docs():
            nonlocal doc_lm_list
            nonlocal total_next

            for doc, lm, src_len in doc_lm_list:
                # as we predict next token we need seq_len + 1
                while len(doc) < self.seq_len + 1:
                    tokens, loss_mask, src = next(token_src)
                    total_next += 1
                    if self.pad:
                        loss_mask[0] = 0
                        reminder = len(tokens) % self.seq_len
                        if reminder != 0:
                            to_pad = self.seq_len - reminder
                            tokens += [0] * to_pad
                            loss_mask += [0] * to_pad
                    doc.extend(tokens)
                    lm.extend(loss_mask)
                    src_len.append((src, len(tokens)))

                assert len(doc) == len(lm)

        def extract_batch():
            nonlocal doc_lm_list
            nonlocal total_tokens

            input_tokens_list = []
            target_tokens_list = []
            loss_mask_list = []

            assert len(doc_lm_list) == self.batch_size

            len_dict = defaultdict(int)
            for i, (doc, lm, src_len) in enumerate(doc_lm_list):
                assert len(doc) == len(lm)
                input_tokens = doc[: self.seq_len]
                target_tokens = doc[1 : self.seq_len + 1]
                loss_mask = lm[1 : self.seq_len + 1]

                doc_lm_list[i][0] = doc[self.seq_len :]
                doc_lm_list[i][1] = lm[self.seq_len :]
                # Logging how much tokens per source we have
                tokens_to_fill = self.seq_len
                src_len_start_idx = 0
                while tokens_to_fill > 0:
                    src = src_len[src_len_start_idx][0]
                    tokens_to_take = min(tokens_to_fill, src_len[src_len_start_idx][1])
                    len_dict[src] += tokens_to_take
                    tokens_to_fill -= tokens_to_take
                    if src_len[src_len_start_idx][1] == tokens_to_take:
                        src_len_start_idx += 1
                    else:
                        assert tokens_to_fill == 0
                        src_len[src_len_start_idx] = (
                            src,
                            src_len[src_len_start_idx][1] - tokens_to_take,
                        )
                doc_lm_list[i][2] = src_len[src_len_start_idx:]

                input_tokens_list.append(input_tokens)
                target_tokens_list.append(target_tokens)
                loss_mask_list.append(loss_mask)

            data_batch = {
                "input_tokens": np.array(input_tokens_list, dtype=np.int32),
                "target_tokens": np.array(target_tokens_list, dtype=np.int32),
                "loss_masks": np.array(loss_mask_list, dtype=np.float32),
            }
            total_tokens += np.prod(data_batch["input_tokens"].shape)
            assert data_batch["input_tokens"].shape == (self.batch_size, self.seq_len)
            assert data_batch["target_tokens"].shape == (self.batch_size, self.seq_len)
            assert data_batch["loss_masks"].shape == (self.batch_size, self.seq_len)

            assert sum(len_dict.values()) == self.batch_size * self.seq_len
            return data_batch, len_dict

        while True:
            populate_docs()
            batch, len_dict = extract_batch()
            len_dict = {f"batch_tokens_per_source/{k}": v for k, v in len_dict.items()}
            yield batch, {
                "dataset_example_index": total_next,
                "dataset_total_tokens": total_tokens,
                **len_dict,
            }


class KPackingDAPipeline(DataPipeline):
    """
    Pipeline that assigns multiple (k) indexes of the batch to a single doc.
    It achieves this by using DocAwareDataPipeline with k times smaller batch
    and k times longer seq_len.
    """

    def __init__(
        self,
        token_source: Iterable[Tuple[List[int], List[float], str]],
        batch_size: int,
        seq_len: int,
        k: int,
        pad: bool = False,
    ) -> None:
        super().__init__(token_source, batch_size, seq_len)
        LOGGER.info(f"KPackingDAPipeline: Batch size {batch_size} k {k}")
        assert batch_size % k == 0
        self.inner_batch_size = batch_size // k
        self.inner_seq_len = k * seq_len
        self.da_pipeline = DocAwareDataPipeline(
            token_source=token_source,
            batch_size=self.inner_batch_size,
            seq_len=self.inner_seq_len,
            pad=pad,
        )

    def __iter__(self):
        data_src = iter(self.da_pipeline)

        while True:
            data_batch, metrics = next(data_src)
            new_data_batch = {}
            for k, v in data_batch.items():
                new_data_batch[k] = v.reshape(self.batch_size, self.seq_len)
            yield new_data_batch, metrics


class TextToToken:
    def __init__(self, text_source: Iterable[Dict[str, str]], text_processor) -> None:
        self.text_source = text_source
        self.text_processor = text_processor

    def __iter__(self):
        for eid, example in enumerate(self.text_source):
            tokens, loss_masks, src = self.text_processor(example)
            yield tokens, loss_masks, src, -1, eid


class TokenFilter:
    """
    Filters out examples that have less than min_example_length tokens.
    Collects statistics about token_source (averaged over num_stat_samples).
    """

    def __init__(
        self,
        token_source: Callable[[], Generator[Tuple[List[int], List[float], str, int, int], None, None]],
        min_example_length: int,
        num_stat_samples: int,
    ) -> None:
        self.token_source = token_source
        self.num_stat_samples = num_stat_samples
        self.min_example_length = min_example_length
        LOGGER.info(f"TokenFilter min_example_length: {min_example_length}")
        self.loc = None
        self.index = None
        self.step_times = []

    def get_metrics(self):
        return {
            "dataset_file_loc": self.loc,
            "dataset_example_index": self.index,
            "dataset_get_time_mean": np.mean(self.step_times),
            "dataset_get_time_max": np.max(self.step_times),
        }

    def __iter__(self):
        last_time = time.time()
        for tokens, loss_mask, source_ds, loc, index in self.token_source():
            if self.min_example_length > 0 and len(tokens) < self.min_example_length:
                continue
            self.loc = loc
            self.index = index
            self.step_times.append(time.time() - last_time)
            if len(self.step_times) > self.num_stat_samples:
                self.step_times = self.step_times[-self.num_stat_samples :]
            yield tokens, loss_mask, source_ds
            last_time = time.time()


LINEAR_PIPELINE = "linear"
DOC_AWARE_PIPELINE = "doc_aware"
DOC_AWARE_PIPELINE_K = "doc_aware_k"

DATA_PIPELINE_CONSTRUCTORS = {
    LINEAR_PIPELINE: LinearPipeline,
    DOC_AWARE_PIPELINE: DocAwareDataPipeline,
    DOC_AWARE_PIPELINE_K: KPackingDAPipeline,
}


def doc_aware_pileline_params(pipeline_str: str):
    """
    Extracts the k from the pipeline_str.
    """
    assert pipeline_str[: len(DOC_AWARE_PIPELINE_K)] == DOC_AWARE_PIPELINE_K
    _, k = pipeline_str.split("_k")
    assert len(k) > 0
    k = int(k)
    return DOC_AWARE_PIPELINE_K, k


def get_dataset_packing(data_pipeline: str):
    """
    DOC_AWARE_PIPELINE_K assigns k batch indices to one doc
    """
    if data_pipeline is not None and data_pipeline[: len(DOC_AWARE_PIPELINE_K)] == DOC_AWARE_PIPELINE_K:
        _, k = doc_aware_pileline_params(data_pipeline)
        return k
    else:
        return 1


def get_data_pipeline_constructor(data_pipeline: str):
    if data_pipeline[: len(DOC_AWARE_PIPELINE_K)] == DOC_AWARE_PIPELINE_K:
        base_name, k = doc_aware_pileline_params(data_pipeline)
        LOGGER.info(f"Using {base_name + str(k)} data pipeline")
        constructor = partial(DATA_PIPELINE_CONSTRUCTORS[base_name], k=k)
        return constructor
    else:
        LOGGER.info(f"Using {data_pipeline} data pipeline")
        return DATA_PIPELINE_CONSTRUCTORS[data_pipeline]
