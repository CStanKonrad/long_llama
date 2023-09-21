import copy
import json
import logging
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import functools
from .arguments import *

LOGGER = logging.Logger("Data Processing", level=logging.INFO)
LOGGER_HANDLER = logging.StreamHandler(sys.stderr)
LOGGER_HANDLER.setFormatter(logging.Formatter("[%(asctime)s] Fine-Tuning [%(levelname)s] : %(message)s"))
LOGGER.addHandler(LOGGER_HANDLER)
IGNORE_INDEX = -100  # pytorch cross_entropy loss ignores tokens with this id


# note how we choose the padding token
# later attention mask will be applied to all padding tokens
def get_padding_token(tokenizer: PreTrainedTokenizer):
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    else:
        return tokenizer.unk_token_id


def handle_data_padding(
    token_arrays: List[np.array], padding_ids: List[int], tokenization_args: TokenizationArgs
) -> List[np.array]:
    """
    For padding the data equally.
    Applies padding only if tokenization_args.always_pad is true.
    Supports two modes of padding:
      * default - add pad tokens on the right side
      * random - sample how many will be added to the right and how many will be added to the left
    Args:
      token_arrays - should be a list of 1D numpy arrays of the same length (for example input_ids and labels)
      padding_ids - values to use for padding
      tokenization_args -
        tokenization_args.always_pad - whether to pad
        tokenization_args.max_total_length - length to pad to (assumes the input is not longer)
        tokenization_args.random_pad - whether to use random or default padding mode
    """

    if len(token_arrays) != len(padding_ids):
        raise ValueError("Number of paddings_ids should match number of token_arrays")

    for ta in token_arrays:
        if len(ta.shape) != 1:
            raise ValueError("token_arrays should be 1D")
        if ta.shape != token_arrays[0].shape:
            raise ValueError("token_arrays should have the same length")

    if tokenization_args.always_pad:
        padding = tokenization_args.max_total_length - token_arrays[0].shape[-1]
        if padding > 0:
            if tokenization_args.random_pad:
                padding_left = np.random.randint(0, padding + 1)
                padding_right = padding - padding_left
            else:
                padding_left = 0
                padding_right = padding

            result = []
            for ta, pi in zip(token_arrays, padding_ids):
                result.append(
                    np.pad(
                        ta,
                        ((padding_left, padding_right),),
                        "constant",
                        constant_values=pi,
                    )
                )
            token_arrays = result

    return token_arrays


def tokenize_text_no_special_tokens(text: str, tokenizer: PreTrainedTokenizer) -> np.array:
    if not isinstance(text, str):
        raise ValueError(f"Expected string got {text}")
    return tokenizer.encode(text, add_special_tokens=False, return_tensors="np")[0].astype(np.int64)


def inst_tuning_data_processor(
    data_args: DataArgs, data, tokenizer: PreTrainedTokenizer, tokenization_args: TokenizationArgs
) -> Dict[str, np.array]:
    def prepare_input_text(
        pre_prompt_text: str,
        prompt_field: Optional[str],
        post_prompt_text: str,
        pre_question_text: str,
        question_field: str,
        post_question_text: str,
        pre_response_text: str,
        response_field: str,
        post_response_text: str,
        data_dict: Dict[str, str],
    ):
        input_data = []

        if prompt_field is not None:
            input_data.append(pre_prompt_text)
            input_data.append(data_dict[prompt_field])
            input_data.append(post_prompt_text)

        if question_field is None:
            raise ValueError("For insturction fine-tuning question_field is required")

        input_data.append(pre_question_text)
        input_data.append(data_dict[question_field])
        input_data.append(post_question_text)

        input_text = "".join(input_data)

        if response_field is None:
            raise ValueError("For insturction fine-tuning response_field is required")

        response_text = "".join([pre_response_text, data_dict[response_field], post_response_text])

        return input_text, response_text

    def tokenize_data(
        input_response: List[Tuple[str, str]],
        tokenizer: PreTrainedTokenizer,
        tokenization_args: TokenizationArgs,
    ) -> List[Dict[str, np.array]]:
        def tokenize_one(input_text: str, response_text: str):
            input_tokens = tokenize_text_no_special_tokens(text=input_text, tokenizer=tokenizer)
            input_tokens = np.pad(input_tokens, ((1, 0),), "constant", constant_values=tokenizer.bos_token_id)

            input_tokens = input_tokens[: tokenization_args.max_input_length]

            response_tokens = tokenize_text_no_special_tokens(text=response_text, tokenizer=tokenizer)
            response_tokens = np.pad(response_tokens, ((0, 1),), "constant", constant_values=tokenizer.eos_token_id)
            response_tokens = response_tokens[: tokenization_args.max_output_length]

            assert len(input_tokens.shape) == 1 and len(response_tokens.shape) == 1
            all_tokens = np.concatenate([input_tokens, response_tokens], axis=-1)

            labels = np.concatenate([np.full_like(input_tokens, IGNORE_INDEX), response_tokens])

            assert labels.shape == all_tokens.shape
            assert len(labels.shape) == 1

            all_tokens = all_tokens[: tokenization_args.max_total_length]
            labels = labels[: tokenization_args.max_total_length]

            all_tokens, labels = handle_data_padding(
                token_arrays=[all_tokens, labels],
                padding_ids=[get_padding_token(tokenizer), IGNORE_INDEX],
                tokenization_args=tokenization_args,
            )

            attention_mask = all_tokens != get_padding_token(tokenizer)

            return all_tokens, labels, attention_mask

        def tokenize_portion(input_response_portion: List[Tuple[str, str]]) -> List[Dict[str, np.array]]:
            result = []
            for input_text, response_text in input_response_portion:
                all_tokens, labels, attention_mask = tokenize_one(input_text, response_text)
                result.append(dict(input_ids=all_tokens, labels=labels, attention_mask=attention_mask))

            return result

        return tokenize_portion(input_response)

    input_response = prepare_input_text(
        pre_prompt_text=data_args.pre_prompt_text,
        prompt_field=data_args.prompt_field,
        post_prompt_text=data_args.post_prompt_text,
        pre_question_text=data_args.pre_question_text,
        question_field=data_args.question_field,
        post_question_text=data_args.post_question_text,
        pre_response_text=data_args.pre_response_text,
        response_field=data_args.response_field,
        post_response_text=data_args.post_response_text,
        data_dict=data,
    )

    elem = tokenize_data([input_response], tokenizer=tokenizer, tokenization_args=tokenization_args)[0]

    return elem


def chat_tuning_data_processor(
    data_args: DataArgs, data, tokenizer: PreTrainedTokenizer, tokenization_args: TokenizationArgs
) -> Dict[str, np.array]:
    input_ids = []
    labels = []
    data = data[data_args.chat_conversations_field]

    model_prefix = tokenize_text_no_special_tokens(text=data_args.chat_model_response_prefix, tokenizer=tokenizer)
    human_prefix = tokenize_text_no_special_tokens(text=data_args.chat_human_response_prefix, tokenizer=tokenizer)

    input_ids.append(tokenize_text_no_special_tokens(data_args.chat_initial_prompt, tokenizer=tokenizer))
    labels.append(np.full_like(input_ids[0], IGNORE_INDEX))

    replace_rules_raw_list = (
        data_args.chat_replace_rules.split("<;>") if data_args.chat_replace_rules is not None else []
    )
    replace_rules_list = []
    for replace_rule in replace_rules_raw_list:
        regex, target = replace_rule.split("<R>")
        LOGGER.debug(f"Chat: Compiling regex {regex} for replacing with {target}")
        regex = re.compile(regex, flags=re.DOTALL)
        replace_rules_list.append((regex, target))

    LOGGER.debug(f"Chat: Compiled {len(replace_rules_list)} replace rules")

    def handle_replacements(text: str) -> str:
        for regex, target in replace_rules_list:
            text = regex.sub(target, text)
        return text

    for part in data:
        text = part[data_args.chat_data_field]

        text = handle_replacements(text)

        tokenized_text = tokenize_text_no_special_tokens(text=text, tokenizer=tokenizer)

        is_model = part[data_args.chat_source_name_field] == data_args.chat_model_source_name
        if is_model:
            token_prefix = model_prefix
            token_sufix = np.array([tokenizer.eos_token_id])
        else:
            token_prefix = human_prefix
            token_sufix = np.empty(0, dtype=np.int64)

        input_ids.append(token_prefix)
        input_ids.append(tokenized_text)
        input_ids.append(token_sufix)

        labels.append(np.full_like(token_prefix, IGNORE_INDEX))
        if is_model:
            labels.append(tokenized_text)
            labels.append(token_sufix)
        else:
            labels.append(np.full_like(tokenized_text, IGNORE_INDEX))
            labels.append(np.full_like(token_sufix, IGNORE_INDEX))

    input_ids = [np.array([tokenizer.bos_token_id])] + input_ids
    input_ids = np.concatenate(input_ids, axis=-1)
    labels = [np.array([IGNORE_INDEX])] + labels
    labels = np.concatenate(labels, axis=-1)

    assert input_ids.shape == labels.shape
    assert len(input_ids.shape) == 1

    input_ids = input_ids[: tokenization_args.max_total_length]
    labels = labels[: tokenization_args.max_total_length]

    input_ids, labels = handle_data_padding(
        token_arrays=[input_ids, labels],
        padding_ids=[get_padding_token(tokenizer), IGNORE_INDEX],
        tokenization_args=tokenization_args,
    )

    attention_mask = input_ids != get_padding_token(tokenizer)
    attention_mask = np.logical_and(attention_mask, input_ids != tokenizer.eos_token_id)

    result = dict(input_ids=input_ids.astype(np.int64), labels=labels.astype(np.int64), attention_mask=attention_mask)
    return result


def get_data_processor(data_args: DataArgs):
    if data_args.data_type == "instructions":
        return inst_tuning_data_processor
    elif data_args.data_type == "chat":
        return chat_tuning_data_processor


def separate_data_args(data_args: DataArgs) -> List[DataArgs]:
    """
    Given the data_args creates a separate instance for each dataset.
    """

    data_types = data_args.data_type.split(",")
    num_datasets = len(data_types)
    data_paths = data_args.data_path.split(",")
    revisions = data_args.data_revision.split(",")
    data_splits = data_args.dataset_split.split(",")

    def split_field(field: Optional[str], dataset_type: str, separator: str, process_field_name_fn):
        if field is None:
            return [None] * num_datasets
        else:
            field_list = field.split(separator)
            if len(field_list) == 1:
                LOGGER.info(f"Broadcastin used for {dataset_type} fields {field_list}.")
                field_list_getter = lambda _: field_list[0]
            else:
                field_list_getter = lambda x: field_list[x]
            result = []
            appended = 0
            for dt in data_types:
                if dt == dataset_type:
                    converted_field_data = process_field_name_fn(field_list_getter(appended))
                    result.append(converted_field_data)
                    appended += 1
                else:
                    result.append(None)
            return result

    def none_str_to_none(x):
        if x == "None":
            return None
        else:
            return x

    split_basic_instruct_field = functools.partial(
        split_field, dataset_type="instructions", separator=",", process_field_name_fn=none_str_to_none
    )

    prompt_fields = split_basic_instruct_field(data_args.prompt_field)
    question_fields = split_basic_instruct_field(data_args.question_field)
    response_fields = split_basic_instruct_field(data_args.response_field)

    split_adv_instruct_field = functools.partial(
        split_field, dataset_type="instructions", separator="<,>", process_field_name_fn=lambda x: x
    )

    pre_prompt_texts = split_adv_instruct_field(data_args.pre_prompt_text)
    post_prompt_texts = split_adv_instruct_field(data_args.post_prompt_text)

    pre_question_texts = split_adv_instruct_field(data_args.pre_question_text)
    post_question_texts = split_adv_instruct_field(data_args.post_question_text)

    pre_response_texts = split_adv_instruct_field(data_args.pre_response_text)
    post_response_texts = split_adv_instruct_field(data_args.post_response_text)

    split_basic_chat_field = functools.partial(
        split_field, dataset_type="chat", separator=",", process_field_name_fn=none_str_to_none
    )

    chat_conversations_fields = split_basic_chat_field(data_args.chat_conversations_field)
    chat_data_fields = split_basic_chat_field(data_args.chat_data_field)
    chat_source_name_fields = split_basic_chat_field(data_args.chat_source_name_field)
    chat_model_source_names = split_basic_chat_field(data_args.chat_model_source_name)

    if data_args.data_filter is None:
        data_filters = [None for _ in range(num_datasets)]
    else:
        data_filters = data_args.data_filter.split("<,>")

    data_proportions = data_args.data_proportions

    LOGGER.info(
        "Praparing configs:\n"
        f"data_types : {data_types}\n"
        f"data_paths : {data_paths}\n"
        f"revisions : {revisions}\n"
        f"data_splits : {data_splits}\n"
        f"data_proportions : {data_proportions}\n"
        f"data_filters : {data_filters}\n"
    )

    if (
        num_datasets != len(data_paths)
        or num_datasets != len(revisions)
        or num_datasets != len(data_splits)
        or num_datasets != len(data_proportions)
        or num_datasets != len(data_filters)
    ):
        raise ValueError(
            "When preparing the mixture provide the same number of elements in: "
            "data_path, data_revision, data_type, data_proportions (separated by ','), data_filters (separated by <,>)"
        )

    splitted_data_args = []
    for d_id in range(num_datasets):
        d_args = copy.deepcopy(data_args)
        d_args.data_type = data_types[d_id]
        d_args.data_path = data_paths[d_id]
        d_args.data_revision = revisions[d_id]
        d_args.dataset_split = data_splits[d_id]
        d_args.data_proportions = data_proportions[d_id]
        d_args.data_filter = data_filters[d_id]

        d_args.prompt_field = prompt_fields[d_id]
        d_args.question_field = question_fields[d_id]
        d_args.response_field = response_fields[d_id]

        d_args.pre_prompt_text = pre_prompt_texts[d_id]
        d_args.post_prompt_text = post_prompt_texts[d_id]

        d_args.pre_question_text = pre_question_texts[d_id]
        d_args.post_question_text = post_question_texts[d_id]

        d_args.pre_response_text = pre_response_texts[d_id]
        d_args.post_response_text = post_response_texts[d_id]

        d_args.chat_conversations_field = chat_conversations_fields[d_id]
        d_args.chat_data_field = chat_data_fields[d_id]
        d_args.chat_source_name_field = chat_source_name_fields[d_id]
        d_args.chat_model_source_name = chat_model_source_names[d_id]
        splitted_data_args.append(d_args)

    return splitted_data_args


def filter_dataset(dataset, data_args: DataArgs, tokenizer: PreTrainedTokenizer):
    """
    For filtering the dataset according to the rules described in data_args.
    data_args should be separated using separate_data_args.
    """

    if data_args.data_filter is not None and data_args.data_filter != "":
        match_mode = "<M>"
        lenlt_mode = "<LENLT>"
        lengt_mode = "<LENGT>"
        toklt_mode = "<TOKLT>"
        tokgt_mode = "<TOKGT>"

        all_modes = [match_mode, lenlt_mode, lengt_mode, toklt_mode, tokgt_mode]

        raw_rules = data_args.data_filter.split("<;>")
        rules = []
        for rr in raw_rules:
            mode_matched = False
            for mode in all_modes:
                if mode in rr:
                    if mode_matched:
                        raise ValueError("Only one mode can be matched")
                    the_trigger = mode
                    mode_matched = True

            if not mode_matched:
                raise ValueError(f"In {rr} at lest one mode must be matched. Modes: {all_modes}")

            field, regex = rr.split(the_trigger)

            LOGGER.info(f"Dataset: Compiling {regex} for mode {the_trigger} with field {field}")
            regex = re.compile(regex, flags=re.DOTALL) if the_trigger == match_mode else int(regex)
            rules.append((field, regex, the_trigger))

        def filtering(x: Dict[str, Any]):
            int_trigger = "<int>"

            def recursive_check_match(regex, v: Union[Dict[str, Any], str], field_nesting: List[str]):
                if len(field_nesting) == 0:
                    assert isinstance(v, str)
                    return regex.match(v) is not None

                fn = field_nesting[0]
                if fn.startswith(int_trigger):
                    fn = fn[len(int_trigger) :]
                    if fn == "*v" or fn == "*^":
                        and_mode = fn == "*^"
                        or_mode = fn == "*v"
                        for elem in v:
                            result = recursive_check_match(regex, elem, field_nesting[1:])
                            if result and or_mode:
                                return True
                            elif (not result) and and_mode:
                                return False
                        return and_mode
                    else:
                        fn = int(fn)
                        return recursive_check_match(regex, v[fn], field_nesting[1:])
                else:
                    return recursive_check_match(regex, v[fn], field_nesting[1:])

            def recursive_count(
                processor: Callable[[str], int], v: Union[Dict[str, Any], str], field_nesting: List[str]
            ):
                if len(field_nesting) == 0:
                    assert isinstance(v, str)
                    return processor(v)
                fn = field_nesting[0]

                if fn.startswith(int_trigger):
                    fn = fn[len(int_trigger) :]
                    if fn == "*":
                        result = 0
                        for elem in v:
                            result += recursive_count(processor, elem, field_nesting[1:])
                        return result
                    else:
                        fn = int(fn)
                        return recursive_count(processor, v[fn], field_nesting[1:])
                else:
                    return recursive_count(processor, v[fn], field_nesting[1:])

            for field, regex, mode in rules:
                field_nesting = field.split(".")
                if mode == match_mode:
                    if not recursive_check_match(regex, x, field_nesting):
                        return False
                elif mode == lenlt_mode:
                    if recursive_count(lambda v: len(v), x, field_nesting) >= regex:
                        return False
                elif mode == lengt_mode:
                    if recursive_count(lambda v: len(v), x, field_nesting) <= regex:
                        return False
                elif mode == toklt_mode:
                    if (
                        recursive_count(
                            lambda v: tokenizer.encode(v, add_special_tokens=False, return_tensors="np").shape[-1],
                            x,
                            field_nesting,
                        )
                        >= regex
                    ):
                        return False
                elif mode == tokgt_mode:
                    if (
                        recursive_count(
                            lambda v: tokenizer.encode(v, add_special_tokens=False, return_tensors="np").shape[-1],
                            x,
                            field_nesting,
                        )
                        <= regex
                    ):
                        return False

            return True

        LOGGER.info(f"Dataset: Compiling {len(rules)} filtering rules")

        return dataset.filter(filtering)
    else:
        return dataset


class SingleTuneDataset(Dataset):
    """
    For handling a single dataset.
    data_args should be separated using separate_data_args.
    """

    def __init__(
        self,
        data_args: DataArgs,
        tokenizer: PreTrainedTokenizer,
        tokenization_args: TokenizationArgs,
        return_pt: bool = True,
        data_processor=None,
    ):
        super().__init__()

        self.data_args = data_args
        self.tokenizer = tokenizer
        self.tokenization_args = tokenization_args
        self.return_pt = return_pt
        self.data_processor = data_processor if data_processor is not None else get_data_processor(data_args=data_args)

        LOGGER.info(
            f"Loading data from {self.data_args.data_path}, revision {self.data_args.data_revision}, split {self.data_args.dataset_split}"
        )
        if self.data_args.data_path is None:
            raise ValueError("No dataset (data_path) specified")
        raw_dataset = load_dataset(
            self.data_args.data_path, revision=self.data_args.data_revision, split=self.data_args.dataset_split
        )

        raw_dataset = filter_dataset(dataset=raw_dataset, data_args=data_args, tokenizer=tokenizer)

        self.raw_data = raw_dataset
        self.length = len(self.raw_data)
        LOGGER.info(f"Single dataset size is {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        raw_data = self.raw_data[i]
        elem = self.data_processor(
            data_args=self.data_args, data=raw_data, tokenizer=self.tokenizer, tokenization_args=self.tokenization_args
        )

        if self.return_pt:
            converter = lambda x: torch.from_numpy(x).to(torch.long)
            converter_mask = lambda x: torch.from_numpy(x).to(torch.bool)
        else:
            converter = lambda x: x
            converter_mask = lambda x: x
        return dict(
            input_ids=converter(elem["input_ids"]),
            labels=converter(elem["labels"]),
            attention_mask=converter_mask(elem["attention_mask"]),
        )


@dataclass
class DatasetProcessingStats:
    ds_name: str
    padding_token_id: int = 0
    processed_tokens: int = 0
    processed_loss_tokens: int = 0
    number_of_get_calls: int = 0

    def update(self, data: Dict[str, Any]):
        def convert_to_numpy(x):
            if isinstance(x, np.ndarray):
                return x
            elif isinstance(x, torch.tensor):
                return x.numpy()
            else:
                raise ValueError("DatasetProcessingStats: Type not supported")

        input_ids = convert_to_numpy(data["input_ids"])
        labels = convert_to_numpy(data["labels"])

        self.processed_tokens += (
            np.logical_and(input_ids != self.padding_token_id, input_ids != IGNORE_INDEX).astype(np.int64).sum().item()
        )
        self.processed_loss_tokens += (
            np.logical_and(labels != self.padding_token_id, labels != IGNORE_INDEX).astype(np.int64).sum().item()
        )
        self.number_of_get_calls += 1


def show_data_stats(data_stats: List[DatasetProcessingStats]):
    total_proc_tokens = 0
    total_proc_loss_tokens = 0
    total_get_calls = 0
    for ds in data_stats:
        total_proc_tokens += ds.processed_tokens
        total_proc_loss_tokens += ds.processed_loss_tokens
        total_get_calls += ds.number_of_get_calls

    result = {}
    for ds in data_stats:
        result[ds.ds_name] = {
            "processed_tokens": ds.processed_tokens,
            "processed_tokens%": 100.0 * ds.processed_tokens / max(total_proc_tokens, 1),
            "processed_loss_tokens": ds.processed_loss_tokens,
            "processed_loss_tokens%": 100.0 * ds.processed_loss_tokens / max(total_proc_loss_tokens, 1),
            "get_calls": ds.number_of_get_calls,
            "get_calls%": ds.number_of_get_calls / max(total_get_calls, 1),
        }

    return json.dumps(result, indent=2)


class MixedTuneDataset(Dataset):
    def __init__(
        self,
        data_args: DataArgs,
        tokenizer: PreTrainedTokenizer,
        tokenization_args: TokenizationArgs,
        return_pt: bool = True,
        mix_seed=42,
        log_stats=False,
    ):
        self.org_data_args = data_args
        self.tokenizer = tokenizer
        self.tokenization_args = tokenization_args
        self.return_pt = return_pt

        data_args = separate_data_args(data_args)
        LOGGER.info(f"Will mix {len(data_args)} datasets")

        self.datasets = []
        total_length = 0
        for da in data_args:
            LOGGER.info(f"Creating dataset with config: {da}")
            ds = SingleTuneDataset(
                data_args=da, tokenizer=tokenizer, tokenization_args=tokenization_args, return_pt=return_pt
            )
            self.datasets.append(ds)
            total_length += len(ds)

        self.total_length = total_length
        LOGGER.info(f"Cumulative dataset size is {self.total_length}")

        if log_stats:
            self.per_dataset_stats = [
                DatasetProcessingStats(
                    ds_name=f"ds{ds_id}={ds.data_args.data_path}",
                    padding_token_id=get_padding_token(tokenizer=tokenizer),
                )
                for ds_id, ds in enumerate(self.datasets)
            ]
        else:
            self.per_dataset_stats = None

        ds_indices = np.arange(len(self.datasets), dtype=np.int32)
        if len(ds_indices) > 1:
            rnd_state = np.random.get_state()
            np.random.seed(mix_seed)
            self.mapping = np.random.choice(
                ds_indices, self.total_length, replace=True, p=self.org_data_args.data_proportions
            )
            np.random.set_state(rnd_state)
        else:
            self.mapping = np.zeros(self.total_length, dtype=np.int32)

    def __len__(self):
        return self.total_length

    def __getitem__(self, i):
        # Not perfect but simple solution
        ds_id = self.mapping[i]
        ds = self.datasets[ds_id]
        i = i % len(ds)
        result = ds[i]

        if self.per_dataset_stats is not None:
            self.per_dataset_stats[ds_id].update(result)
            LOGGER.info(show_data_stats(self.per_dataset_stats))

        return result


class DataCollator:
    tokenizer: PreTrainedTokenizer

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, inputs):
        input_ids, labels, attention_masks = [], [], []
        for elem in inputs:
            input_ids.append(elem["input_ids"])
            labels.append(elem["labels"])
            attention_masks.append(elem["attention_mask"])

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=get_padding_token(self.tokenizer)
        )

        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=False)

        return dict(input_ids=input_ids, labels=labels, attention_mask=attention_masks)
