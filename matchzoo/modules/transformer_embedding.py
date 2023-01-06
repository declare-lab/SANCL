from torch import nn
from matchzoo.helper import logger
from transformers import (
    XLMConfig,
    XLMModel,
    XLMRobertaConfig,
    XLMRobertaModel,
    BertConfig,
    BertModel
)

MODEL_CLASS = {
    'mbert': (BertConfig, BertModel),
    'xlm': (XLMConfig, XLMModel),
    'xlm-r': (XLMRobertaConfig, XLMRobertaModel)
}


class TransformerEmbeddingLayer(nn.Module):
    """
    Transformer Based Embedding Layer
    """

    def __init__(self, embedding_type, config_dir, pretrained_file, stage: str = 'train'):
        super().__init__()
        model_config, model = MODEL_CLASS[embedding_type]
        self.model_config = model_config.from_pretrained(config_dir)

        if stage == 'train':
            logger.info("Loading pretrained transformer embedding model from %s ..." % pretrained_file)
            self.model = model.from_pretrained(
                pretrained_file,
                config=self.model_config,
                local_files_only=True
            )
        else:
            logger.info("Directly initialize transformer embedding model ...")
            self.model = model(self.model_config)

    def forward(self, input_indices, input_mask, input_types, input_lang=None):
        if input_lang is None:
            outputs = self.model(
                input_ids=input_indices,
                attention_mask=input_mask,
                token_type_ids=input_types
            )
        else:
            raise NotImplementedError
        return outputs[0]
