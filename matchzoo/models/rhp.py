#!/usr/bin/env python
# coding:utf-8

import torch

from torch import nn
from matchzoo.modules.rnn import LSTM
from matchzoo.modules.embedding_layer import EmbeddingLayer
from matchzoo.modules.utils import generate_seq_mask
from matchzoo.modules.transformer_embedding import TransformerEmbeddingLayer
from matchzoo.pipeline.rhp_pipeline import RHPPipeline
from matchzoo.pipeline.trans_rhp_pipeline import TransformersRHPPipeline


class PREncoder(nn.Module):
    def __init__(self, config, input_dim):
        super(PREncoder, self).__init__()
        self.config = config
        self.rnn = LSTM(
            layers=config.text_encoder.RNN.num_layers,
            input_dim=input_dim,
            output_dim=config.text_encoder.RNN.hidden_dimension,
            batch_first=True,
            bidirectional=config.text_encoder.RNN.bidirectional
        )
        hidden_dimension = config.text_encoder.RNN.hidden_dimension
        if config.text_encoder.RNN.bidirectional:
            hidden_dimension *= 2
        self.hidden_dimension = hidden_dimension
        self.rnn_dropout = torch.nn.Dropout(p=config.text_encoder.RNN.dropout)

    def forward(self, inputs, seq_lens):
        """
        :param inputs: torch.FloatTensor, embedding, (batch, max_len, embedding_dim)
        :param seq_lens: torch.LongTensor, (batch, max_len)
        :return:
        """
        text_output, _ = self.rnn(inputs, seq_lens)
        text_output = self.rnn_dropout(text_output)
        text_output = text_output.transpose(1, 2)
        return text_output


class ProductAwareAttention(nn.Module):
    def __init__(self, config):
        super(ProductAwareAttention, self).__init__()

        hidden_dimension = config.text_encoder.RNN.hidden_dimension
        if config.text_encoder.RNN.bidirectional:
            hidden_dimension *= 2

        self.w = nn.Parameter(torch.randn(hidden_dimension, hidden_dimension))
        self.b = nn.Parameter(torch.randn(1, 1, hidden_dimension))

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.b)

    def forward(self,
                product_repr,
                product_seq_lens,
                review_repr,
                review_seq_lens):
        '''
        :param product_repr: torch.FloatTensor (batch, hidden_dimension, product_seq_lens)
        :param product_seq_lens: torch.LongTensor, (batch, max_len)
        :param review_repr: torch.FloatTensor (batch, hidden_dimension, review_seq_lens)
        :param review_seq_lens: torch.LongTensor, (batch, max_len)
        '''

        # (batch, product_seq_lens, hidden_dimension)
        p = torch.matmul(product_repr.transpose(1, 2), self.w)
        p = p + self.b
        p = torch.relu(p)  # (batch, product_seq_lens, hidden_dimension)
        # (batch, product_seq_lens, review_seq_lens)
        q = torch.matmul(p, review_repr)

        # (batch, product_seq_lens)
        p_mask = generate_seq_mask(product_seq_lens)
        p_mask = p_mask.unsqueeze(-1)  # (batch, product_seq_lens, 1)
        q = q * p_mask.float() + (~p_mask).float() * (-1e23)
        q = torch.softmax(q, dim=1)

        r_add = torch.matmul(product_repr, q)
        r = r_add + review_repr   # (batch, hidden_dimension, review_seq_lens)

        r = r.transpose(1, 2)  # (batch, review_seq_lens, hidden_dimension)
        r_mask = generate_seq_mask(review_seq_lens)  # (batch, review_seq_lens)
        r_mask = r_mask.unsqueeze(-1)
        r = r * r_mask.float()  # (batch, review_seq_lens, hidden_dimension)
        return r


class RHPNet(nn.Module):
    def __init__(self, config, pipeline: RHPPipeline, stage: str):
        super(RHPNet, self).__init__()
        self.config = config

        self.product_rnn = PREncoder(
            config, config.embedding.product_token.dimension)
        self.review_rnn = PREncoder(
            config, config.embedding.review_token.dimension)
        self.pr_aware_attn = ProductAwareAttention(config)

        prd_vocab = pipeline.prd_text_field.vocab
        rvw_vocab = pipeline.rvw_text_field.vocab
        prd_map = prd_vocab.v2i
        rvw_map = rvw_vocab.v2i

        self.product_token_embedding = EmbeddingLayer(
            vocab_map=prd_map,
            embedding_dim=config.embedding.product_token.dimension,
            vocab_name='product_token',
            dropout=config.embedding.product_token.dropout,
            embed_type=config.embedding.product_token.type,
            padding_index=prd_vocab.pad_index,
            pretrained_dir=config.embedding.product_token.pretrained_file,
            stage=stage,
            initial_type=config.embedding.product_token.init_type
        )

        self.review_token_embedding = EmbeddingLayer(
            vocab_map=rvw_map,
            embedding_dim=config.embedding.review_token.dimension,
            vocab_name='review_token',
            dropout=config.embedding.review_token.dropout,
            embed_type=config.embedding.review_token.type,
            padding_index=rvw_vocab.pad_index,
            pretrained_dir=config.embedding.review_token.pretrained_file,
            stage=stage,
            initial_type=config.embedding.review_token.init_type
        )

        self.linear = nn.Linear(self.review_rnn.hidden_dimension, 1)

    def forward(self, batch):
        # get distributed representation of tokens, (batch_size, max_length, embedding_dimension)
        product_embedding = self.product_token_embedding(
            batch['text_left'])
        review_embedding = self.review_token_embedding(batch['text_right'])

        # get the length of sequences for dynamic rnn, (batch_size, 1)
        product_seq_len = batch['text_left_length']
        review_seq_len = batch['text_right_length']

        product_repr = self.product_rnn(product_embedding, product_seq_len)
        review_repr = self.review_rnn(review_embedding, review_seq_len)

        review_repr = self.pr_aware_attn(product_repr,
                                         product_seq_len,
                                         review_repr,
                                         review_seq_len)

        review_repr = review_repr.sum(dim=1) / review_seq_len.unsqueeze(-1)

        logits = self.linear(review_repr)
        return logits


class TransformerBasedProductAwareAttention(nn.Module):
    def __init__(self, hidden_dimension):
        super().__init__()
        self.w = nn.Parameter(torch.randn(hidden_dimension, hidden_dimension))
        self.b = nn.Parameter(torch.randn(1, 1, hidden_dimension))

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.b)

    def forward(self,
                product_repr,
                product_seq_lens,
                review_repr,
                review_seq_lens):
        '''
        :param product_repr: torch.FloatTensor (batch, hidden_dimension, product_seq_lens)
        :param product_seq_lens: torch.LongTensor, (batch, max_len)
        :param review_repr: torch.FloatTensor (batch, hidden_dimension, review_seq_lens)
        :param review_seq_lens: torch.LongTensor, (batch, max_len)
        '''

        # (batch, product_seq_lens, hidden_dimension)
        p = torch.matmul(product_repr.transpose(1, 2), self.w)
        p = p + self.b
        p = torch.relu(p)  # (batch, product_seq_lens, hidden_dimension)
        # (batch, product_seq_lens, review_seq_lens)
        q = torch.matmul(p, review_repr)

        # (batch, product_seq_lens)
        p_mask = generate_seq_mask(product_seq_lens)
        p_mask = p_mask.unsqueeze(-1)  # (batch, product_seq_lens, 1)
        q = q * p_mask.float() + (~p_mask).float() * (-1e23)
        q = torch.softmax(q, dim=1)

        r_add = torch.matmul(product_repr, q)
        r = r_add + review_repr   # (batch, hidden_dimension, review_seq_lens)

        r = r.transpose(1, 2)  # (batch, review_seq_lens, hidden_dimension)
        r_mask = generate_seq_mask(review_seq_lens)  # (batch, review_seq_lens)
        r_mask = r_mask.unsqueeze(-1)
        r = r * r_mask.float()  # (batch, review_seq_lens, hidden_dimension)
        return r


class TransformerBasedRHPNet(nn.Module):
    def __init__(self, config, pipeline: TransformersRHPPipeline, stage: str):
        super().__init__()
        self.config = config
        self.embedding_layer = TransformerEmbeddingLayer(
            config.embedding.type,
            config.embedding.config_dir,
            config.embedding.pretrained_file,
            stage=stage
        )

        self.product_rnn = PREncoder(
            config, config.embedding.hidden_dimension)
        self.review_rnn = PREncoder(
            config, config.embedding.hidden_dimension)

        hidden_dimension = self.product_rnn.hidden_dimension
        self.pr_aware_attn = TransformerBasedProductAwareAttention(hidden_dimension)
        self.linear = nn.Linear(hidden_dimension, 1)

    def forward(self, batch):
        product_seq_len = batch['text_left_length']
        review_seq_len = batch['text_right_length']

        # print(batch['text_left_input_ids'])
        # print(batch['text_left_attention_mask'])
        # print(batch['text_left_token_type_ids'])
        with torch.no_grad():
            product_embed = self.embedding_layer(
                input_indices=batch['text_left_input_ids'],
                input_mask=batch['text_left_attention_mask'],
                input_types=batch['text_left_token_type_ids']
            )
            review_embed = self.embedding_layer(
                input_indices=batch['text_right_input_ids'],
                input_mask=batch['text_right_attention_mask'],
                input_types=batch['text_right_token_type_ids']
            )
            # print(product_repr.size())
            # print(review_repr.size())

        product_repr = self.product_rnn(product_embed, product_seq_len)
        review_repr = self.review_rnn(review_embed, review_seq_len)

        # print(review_seq_len)
        review_repr = self.pr_aware_attn(product_repr,
                                         product_seq_len,
                                         review_repr,
                                         review_seq_len)

        review_repr = review_repr.mean(dim=1)
        logits = self.linear(review_repr)
        return logits


class PHP(nn.Module):
    def __init__(self, config, pipeline: RHPPipeline, stage: str):
        super().__init__()
        self.config = config

        prd_vocab = pipeline.prd_text_field.vocab
        rvw_vocab = pipeline.rvw_text_field.vocab
        prd_map = prd_vocab.v2i
        rvw_map = rvw_vocab.v2i

        self.product_token_embedding = EmbeddingLayer(
            vocab_map=prd_map,
            embedding_dim=config.embedding.product_token.dimension,
            vocab_name='product_token',
            dropout=config.embedding.product_token.dropout,
            embed_type=config.embedding.product_token.type,
            padding_index=prd_vocab.pad_index,
            pretrained_dir=config.embedding.product_token.pretrained_file,
            stage=stage,
            initial_type=config.embedding.product_token.init_type
        )

        self.review_token_embedding = EmbeddingLayer(
            vocab_map=rvw_map,
            embedding_dim=config.embedding.review_token.dimension,
            vocab_name='review_token',
            dropout=config.embedding.review_token.dropout,
            embed_type=config.embedding.review_token.type,
            padding_index=rvw_vocab.pad_index,
            pretrained_dir=config.embedding.review_token.pretrained_file,
            stage=stage,
            initial_type=config.embedding.review_token.init_type
        )
        self.product_rnn = PREncoder(
            config, config.embedding.product_token.dimension)
        self.review_rnn = PREncoder(
            config, config.embedding.review_token.dimension)

        self.linear = nn.Linear(self.review_rnn.hidden_dimension, 1)

    def forward(self, batch):
        # get distributed representation of tokens, (batch_size, max_length, embedding_dimension)
        product_embedding = self.product_token_embedding(
            batch['text_left'])
        review_embedding = self.review_token_embedding(batch['text_right'])

        # get the length of sequences for dynamic rnn, (batch_size, 1)
        product_seq_len = batch['text_left_length']
        review_seq_len = batch['text_right_length']

        product_repr = self.product_rnn(product_embedding, product_seq_len)
        review_repr = self.review_rnn(review_embedding, review_seq_len)

        product_repr = product_repr.mean(-1)
        review_repr = review_repr.mean(-1)

        # revalance
        rev = product_repr * review_repr
        logits = self.linear(rev)
        return logits


class EGCNN(nn.Module):
    def __init__(self, config, pipeline: RHPPipeline, stage: str):
        super().__init__()
        self.config = config

        prd_vocab = pipeline.prd_text_field.vocab
        rvw_vocab = pipeline.rvw_text_field.vocab
        prd_map = prd_vocab.v2i
        rvw_map = rvw_vocab.v2i

        self.product_token_embedding = EmbeddingLayer(
            vocab_map=prd_map,
            embedding_dim=config.embedding.product_token.dimension,
            vocab_name='product_token',
            dropout=config.embedding.product_token.dropout,
            embed_type=config.embedding.product_token.type,
            padding_index=prd_vocab.pad_index,
            pretrained_dir=config.embedding.product_token.pretrained_file,
            stage=stage,
            initial_type=config.embedding.product_token.init_type
        )

        self.review_token_embedding = EmbeddingLayer(
            vocab_map=rvw_map,
            embedding_dim=config.embedding.review_token.dimension,
            vocab_name='review_token',
            dropout=config.embedding.review_token.dropout,
            embed_type=config.embedding.review_token.type,
            padding_index=rvw_vocab.pad_index,
            pretrained_dir=config.embedding.review_token.pretrained_file,
            stage=stage,
            initial_type=config.embedding.review_token.init_type
        )

        self.emb_linear = nn.Sequential(
            nn.Linear(config.embedding.review_token.dimension, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Conv1d(config.embedding.review_token.dimension, 64, 5)
        self.linear = nn.Linear(64, 1)

    def forward(self, batch):
        # get distributed representation of tokens, (batch_size, max_length, embedding_dimension)
        product_embedding = self.product_token_embedding(batch['text_left'])
        review_embedding = self.review_token_embedding(batch['text_right'])

        # filtering results
        sent_embedding = torch.cat((product_embedding, review_embedding), dim=1)
        attention = self.emb_linear(sent_embedding)
        attended_embedding = attention * sent_embedding

        # conv
        embedding = self.conv(attended_embedding.transpose(1, 2))
        pooling = embedding.max(dim=-1)[0]
        logits = self.linear(pooling)
        return logits
