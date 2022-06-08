import json
from os.path import join
from typing import Optional

from dataclasses import dataclass, field, asdict


@dataclass
class _Args:
    def to_dict(self):
        return asdict(self)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ModelConstructArgs(_Args):
    model_type: str = field(metadata={"help": "Pretrained model path"})
    head_type: str = field(metadata={"choices": ["linear", "linear_nested", "crf", "crf_nested"], "help": "Type of head"})
    use_word: bool = field(metadata={"help": "whether to use word concatenated with char"})
    resumed_training: bool = field(metadata={'help': "whether to resume from existing checkpoint"})
    model_path: Optional[str] = field(default=None, metadata={"help": "Pretrained model path"})
    word_model_path: Optional[str] = field(default=None, metadata={'help': "Pretrained word-level model path"})
    init_model: Optional[int] = field(default=0, metadata={"choices": [0, 1], "help": "Init models' parameters"})

@dataclass
class CBLUEDataArgs(_Args):
    cblue_root: str = field(metadata={"help": "CBLUE data root"})
    max_length: Optional[int] = field(default=128, metadata={"help": "Max sequence length"})

