from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelArgs:
    model_path: str = field(default="syzymon/long_llama_3b_v1_1")
    mem_dtype: str = field(default="bfloat16")
    last_context_length: int = field(default=1024)
    torch_attention: bool = field(default=False)
    torch_dtype: str = field(default="float32")
    gradient_checkpoint_every_ith: int = field(default=1)


@dataclass
class DataArgs:
    data_type: str = field(
        default="instructions",
        metadata={
            "help": """
                `,` separated list indicating the type of each dataset.
                Available types: 'instructions' and 'chat'.
                Examples:
                  * 'instructions'
                  * 'instructions,chat' - the first dataset is for instruction tuning
                    whereas the second one is for chat tuning.
                """
        },
    )  # instructions or chat

    data_filter: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                """
                '<,>' separated list of '<;>' separated rules of format field_name<M>regex for filtering out the data.
                For example 'lang<M>en<;>conversations.<int>*^.source<M>gpt<,>lang<M>pl'
                will take from the first dataset the records such that their field lang matches the regex en
                and the following property holds: when we take the conversation field and look at all its elements
                then each of them has a field source that matches the regex gpt.
                From the second dataset, it will take the records with field lang matching regex pl.
                Consider another example: '<,>lang<M>^en$<;>conversations.<int>*^.value<M>(?i)^((?!openai).)*$<;>conversations.<int>*^.value<M>^((?!DAN).)*$<;>conversations.<int>0.value<LENLT>8000'.
                Here we do not filter the data coming from the first dataset.
                From the second dataset we take the records such that:
                  * field lang is equal to 'en'
                  * the conversations mention neither openai nor DAN.
                  * the first part of the conversation has at most 8000 chars
                """
            )
        },
    )

    data_path: str = field(
        default=None,
        metadata={
            "help": """
                    Hugging Face dataset(s) name/path; separator ','
                    Examples: 
                        * Open-Orca/OpenOrca
                        * 'Open-Orca/OpenOrca,zetavg/ShareGPT-Processed'
                    """
        },
    )
    data_revision: Optional[str] = field(
        default=None,
        metadata={
            "help": """
                    Revision for each Hugging Face dataset; separator ','
                    Examples:
                        * 'f0823c7ffc48c9d33a42f16cf0b885fed4a7d0a1'
                        * 'f0823c7ffc48c9d33a42f16cf0b885fed4a7d0a1,15968d6dfa02529988da12e382af3ab7c857ebcd'
                    """
        },
    )
    dataset_split: str = field(
        default="train",
        metadata={
            "help": """
                    Split for each Hugging Face dataset; separator ','
                    Examples:
                        * 'train'
                        * 'train,train'
                    """
        },
    )

    # instructions
    pre_prompt_text: str = field(default="")
    prompt_field: Optional[str] = field(default=None)
    post_prompt_text: str = field(default="\n")

    pre_question_text: str = field(default="")
    question_field: str = field(default=None)
    post_question_text: str = field(default="\n")

    pre_response_text: str = field(default="")
    response_field: str = field(default=None)
    post_response_text: str = field(default="")

    # chat
    chat_conversations_field: str = field(
        default="conversations", metadata={"help": "Name of the field with conversations list"}
    )
    chat_data_field: str = field(default="value", metadata={"help": "Name of field with text"})
    chat_source_name_field: str = field(
        default="from", metadata={"help": "Name of field describing the source (human/ai) of the text"}
    )
    chat_model_source_name: str = field(
        default="gpt", metadata={"help": "Name of the text source that should be used to tune the model"}
    )
    chat_initial_prompt: str = field(default="You are a helpful ASSISTANT.\n\n")
    chat_replace_rules: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "'<;>' separated list o describing pairs of replace regular expressions"
                "for example, 'a<R>b<;>c<R>d' means first replace text that matches regex 'a' with string 'b'"
                "then do the same for 'c' and 'd'."
            )
        },
    )

    chat_model_response_prefix: str = field(default="\nASSISTANT: ")
    chat_human_response_prefix: str = field(default="\nUSER: ")

    # proportions (for mixed dataset)
    data_proportions: List[float] = field(
        default_factory=lambda: [1.0], metadata={"help": "Space separated probability of sampling (for each dataset)"}
    )


@dataclass
class TokenizationArgs:
    # Note that max_input_length and max_output_length are only used for instructions data (not for chat)
    # max_total_length is used for both
    max_input_length: int = field(default=2048)
    max_output_length: int = field(default=2048)
    max_total_length: int = field(default=2048)
    always_pad: bool = field(default=True, metadata={"help": "Whether to always pad data to max_total_length tokens"})
    random_pad: bool = field(
        default=True,
        metadata={
            "help": "Whether add padding tokens to the right only or sample the amount of left and right padding"
        },
    )
