"""Basic Preprocessor."""
from matchzoo import DataPack
from matchzoo.engine.base_pipeline import BasePipeline
from matchzoo.preprocessors import TransformersTextFieldPreprocessor


class TransformersRHPPipeline(BasePipeline):
    def __init__(self,
                 tokenizer_type,
                 pretrained_dir,
                 do_lower_case: bool,
                 language: str,):
        """Initialization."""
        self.text_field = TransformersTextFieldPreprocessor(
            tokenizer_type,
            pretrained_dir,
            do_lower_case,
            language,
            field=['text_left', 'text_right'],
            mode=['left', 'right']
        )

    def fit(self, data_pack: DataPack, verbose: int = 1):
        self.text_field.fit(data_pack, verbose)
        return self

    def transform(self, data_pack: DataPack, verbose: int = 1) -> DataPack:
        """
        Apply transformation on data, create truncated length representation.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = self.text_field.transform(data_pack, verbose=verbose)
        return data_pack

    def save(self, save_to):
        self.text_field.save(save_to)

    def load(self, load_from):
        self.text_field.load(load_from)

    def info(self):
        return "Using Transformers Tokenizer"
