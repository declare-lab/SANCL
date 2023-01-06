import typing

from pathlib import Path
from cytoolz.curried import partial
from matchzoo import DataPack
from .text_field_preprocessor import TextFieldPreprocessor
from transformers import (
    BertTokenizer,
    XLMTokenizer,
    XLMRobertaTokenizer
)
from matchzoo.helper import logger

TOKENIZER_CLASS = {
    'mbert': BertTokenizer,
    'xlm': XLMTokenizer,
    'xlm-r': XLMRobertaTokenizer
}


class TransformersTextFieldPreprocessor(TextFieldPreprocessor):
    DEFAULT_SUFFIX = 'tokenizer'

    def __init__(
        self,
        tokenizer_type,
        pretrained_dir,
        do_lower_case: bool,
        language: str,
        field: typing.Union[str, typing.List[str]],
        mode: typing.Union[str, typing.List[str]]
    ):
        super().__init__(field, mode)
        self.tokenizer_type = TOKENIZER_CLASS[tokenizer_type]
        self.pretrained_dir = pretrained_dir
        self.do_lower_case = do_lower_case
        self.language = language

    def _build_unit(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def save(self, save_to):
        save_to = Path(save_to)
        save_to = save_to.joinpath(self.DEFAULT_SUFFIX)
        self.tokenizer.save_pretrained(save_to)

    def load(self, load_from):
        load_from = Path(load_from)
        load_from = load_from.joinpath(self.DEFAULT_SUFFIX)
        self.tokenizer = self.tokenizer_type.from_pretrained(str(load_from))

    def fit(self, data_pack: DataPack, verbose: int = 1):
        """
        Initialize the tokenize setting.
        """
        logger.info('Loading %s from pretrained dir %s...' % (
            self.tokenizer_type.__name__,
            self.pretrained_dir
        ))

        self.tokenizer = self.tokenizer_type.from_pretrained(
            self.pretrained_dir,
            do_lowercase_and_remove_accent=self.do_lower_case
        )
        return self

    def transform(self, data_pack: DataPack, verbose: int = 1) -> DataPack:
        """
        Apply transformation on data, create truncated length representation.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()
        for f, m in self._field_w_mode:
            data_pack.apply_on_field(self.tokenize, field=f, mode=m,
                                     inplace=True, verbose=verbose)

            data_pack.append_field_length(
                field=f, mode=m, inplace=True, verbose=verbose)
            data_pack.drop_field_empty(field=f, mode=m, inplace=True)

        return data_pack
    
    def get_index(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    @property
    def pad_index(self):
        return self.get_index(self.tokenizer.special_tokens_map['pad_token'])

    @property
    def cls_index(self):
        return self.get_index(self.tokenizer.special_tokens_map['cls_token'])

    @property
    def sep_index(self):
        return self.get_index(self.tokenizer.special_tokens_map['sep_token'])

    @property
    def lang_index(self):
        if hasattr(self.tokenizer, 'lang2id'):
            return self.tokenizer.lang2id[self.language]
        return None

    def tokenize(self, text):
        import ipdb; ipdb.set_trace()
        return self.tokenizer.tokenize(text)
