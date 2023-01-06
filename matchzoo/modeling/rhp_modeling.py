import os
import matchzoo as mz

from collections import defaultdict
from matchzoo.dataloader import InstanceDataset
from matchzoo.dataloader.callbacks.load_img import LoadImage
from matchzoo.dataloader.callbacks.padding import (TextFieldPadding, ImageFieldPadding)

from matchzoo.models.rhp import (
    EGCNN,
    PHP
)
from matchzoo.helper import logger
from matchzoo.modeling import BasicModeling
from matchzoo.models.bimpm import BiMPM
from matchzoo.models.conv_knrm import ConvKNRM
from matchzoo.models.rhp import RHPNet
from matchzoo.pipeline.rhp_pipeline import RHPPipeline


class PRH(BasicModeling):
    '''
    A modeling implementation with Review helpfulness prediction.
    '''
    def build_metrics(self):
        threshold = self.config.test.get('threshold', 0)
        self.metrics = [
            mz.metrics.MeanReciprocalRank(threshold=threshold),
            mz.metrics.MeanAveragePrecision(threshold=threshold),
            mz.metrics.NormalizedDiscountedCumulativeGain(k=3, threshold=threshold),
            mz.metrics.NormalizedDiscountedCumulativeGain(k=5, threshold=threshold)
        ]

    def build_preprocessor(self):
        self.preprocessor = RHPPipeline(
            language=self.config.data.language,
            prd_filter_low_freq=self.config.embedding.product_token.filter_low_freq,
            rvw_filter_low_freq=self.config.embedding.review_token.filter_low_freq
        )

    def build_model(self):
        self.model = RHPNet(self.config, self.preprocessor, self.stage)

    def load_data(self, dataclass):
        train_pack = dataclass.load_data(
            read_type=self.config.data.read_type,
            feature_root=self.config.data.feat_dir,
            cat=self.config.data.cat,
            data_root=self.config.data.data_dir,
            stage='train',
            task=self.task
        )

        dev_pack = dataclass.load_data(
            read_type=self.config.data.read_type,
            feature_root=self.config.data.feat_dir,
            cat=self.config.data.cat,
            data_root=self.config.data.data_dir,
            stage='dev',
            task=self.task
        )

        test_pack = dataclass.load_data(
            read_type=self.config.data.read_type,
            feature_root=self.config.data.feat_dir,
            cat=self.config.data.cat,
            data_root=self.config.data.data_dir,
            stage='test',
            task=self.task
        )

        return (train_pack, dev_pack, test_pack)

    def build_dataset_callback(self):
        self.dataset_callback = defaultdict(list)
        if self.config.contains('input_setting') and self.config.input_setting.get('use_image', False):
            for k in self.datapack:
                image_loading = LoadImage(
                    feat_dir=os.path.join(self.config.data.feat_dir, k),
                    feat_size=self.config.input_setting.image_input_dim,
                    max_roi_num=self.config.input_setting.max_roi_num,
                    img_min_length=self.config.input_setting.img_min_length
                )
                self.dataset_callback[k].append(image_loading)

    def build_dataloader_callback(self):
        callbacks = []
        padding = TextFieldPadding(
            text_fields=['text_left', 'text_right'],
            fixed_length=[self.config.text_encoder.max_length] * 2,
            pad_word_value=[
                self.preprocessor.prd_text_field.vocab.pad_index,
                self.preprocessor.rvw_text_field.vocab.pad_index
            ],
            pad_word_mode='post'
        )
        callbacks.append(padding)

        if self.config.contains('input_setting') and self.config.input_setting.get('use_image', False):
            image_padding = ImageFieldPadding(
                image_fields=['image_left', 'image_right'],
                max_roi_per_inst=self.config.input_setting.max_roi_num,
                feat_size=self.config.input_setting.image_input_dim,
                fixed_length=None,
                max_length=self.config.input_setting.img_max_length,
                min_length=self.config.input_setting.img_min_length,
                pad_word_mode='post'
            )
            callbacks.append(image_padding)
        self.dataloader_callback = callbacks


class BigDataPRH(PRH):
    def build_dataset(self):
        for k, v in self.datapack.items():
            if v is None:
                continue

            if k == 'train':
                logger.info('Build the %s dataset with %d batch size...' % (
                    k, self.config.train.batch_size * self.config.train.allocate_num
                )
                )
                dataset = InstanceDataset(
                    data_pack=self.datapack[k],
                    mode=self.config.train.mode,
                    num_dup=self.config.train.num_dup,
                    num_neg=self.config.train.num_neg,
                    shuffle=self.config.train.shuffle,
                    allocate_num=self.config.train.allocate_num,
                    resample=self.config.train.resample,
                    callbacks=self.dataset_callback.get(k, None),
                    weighted_sampling=self.config.train.weighted_sampling,
                    relation_building_interval=self.config.train.relation_building_interval,
                    relation_checkpoint=self.config.train.relation_checkpoint if self.config.train.contains('relation_checkpoint') else None
                )
            else:
                logger.info('Build the %s dataset with %d batch size...' % (
                    k, self.config.test.batch_size * self.config.test.allocate_num
                )
                )
                # in order to quickly evaluate the method when using `dev` stage
                if k == 'dev':
                    dataset = InstanceDataset(
                        data_pack=self.datapack[k],
                        mode='pair',
                        num_dup=self.config.eval.num_dup,
                        num_neg=self.config.eval.num_neg,
                        max_pos_samples=self.config.eval.max_pos_samples,
                        shuffle=False,
                        allocate_num=self.config.eval.allocate_num,
                        callbacks=self.dataset_callback.get(k, None),
                        resample=self.config.eval.resample,
                        weighted_sampling=self.config.eval.weighted_sampling,
                        relation_building_interval=self.config.eval.relation_building_interval
                    )
                else:
                    dataset = InstanceDataset(
                        data_pack=self.datapack[k],
                        mode='point',
                        shuffle=False,
                        allocate_num=self.config.test.allocate_num,
                        callbacks=self.dataset_callback.get(k, None)
                    )
            self.dataset[k] = dataset


class ConvKNRMRHP(PRH):
    def build_model(self):
        self.model = ConvKNRM(self.config, self.preprocessor, self.stage)
    
    def build_preprocessor(self):
        self.preprocessor = RHPPipeline(
            language=self.config.data.language,
            prd_filter_low_freq=self.config.preprocess.prd_filter_low_freq,
            rvw_filter_low_freq=self.config.preprocess.rvw_filter_low_freq
        )

    def build_dataloader_callback(self):
        padding = TextFieldPadding(
            text_fields=['text_left', 'text_right'],
            fixed_length=[self.config.input_setting.max_length] * 2,
            pad_word_value=[
                self.preprocessor.prd_text_field.vocab.pad_index,
                self.preprocessor.rvw_text_field.vocab.pad_index
            ],
            pad_word_mode='post'
        )
        self.dataloader_callback = [padding]


class BigDataConvKNRMRHP(BigDataPRH):
    def build_model(self):
        self.model = ConvKNRM(self.config, self.preprocessor, self.stage)
    
    def build_preprocessor(self):
        self.preprocessor = RHPPipeline(
            language=self.config.data.language,
            prd_filter_low_freq=self.config.preprocess.prd_filter_low_freq,
            rvw_filter_low_freq=self.config.preprocess.rvw_filter_low_freq
        )

    def build_dataloader_callback(self):
        padding = TextFieldPadding(
            text_fields=['text_left', 'text_right'],
            fixed_length=[self.config.input_setting.max_length] * 2,
            pad_word_value=[
                self.preprocessor.prd_text_field.vocab.pad_index,
                self.preprocessor.rvw_text_field.vocab.pad_index
            ],
            pad_word_mode='post'
        )
        self.dataloader_callback = [padding]


class BiMPMRHP(ConvKNRMRHP):
    def build_model(self):
        self.model = BiMPM(self.config, self.preprocessor, self.stage)


class EGCNNRHP(PRH):
    def build_model(self):
        self.model = EGCNN(self.config, self.preprocessor, self.stage)
