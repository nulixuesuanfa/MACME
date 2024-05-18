"""VSE modules"""

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from lib.modules.resnet import ResnetFeatureExtractor
from lib.modules.aggr.gpo import GPO
from lib.modules.mlp import MLP

import logging
import json
import os
import torchtext

logger = logging.getLogger(__name__)


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


def get_text_encoder(vocab_size, embed_size, word_dim, num_layers, opt,use_bi_gru=True, no_txtnorm=False):
    return EncoderText(vocab_size, embed_size, word_dim, num_layers,opt, use_bi_gru=use_bi_gru,
                       no_txtnorm=no_txtnorm)


def get_image_encoder(img_dim, embed_size, precomp_enc_type='basic',
                      backbone_source=None, backbone_path=None, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImageAggr(
            img_dim, embed_size, precomp_enc_type, no_imgnorm)
    elif precomp_enc_type == 'backbone':
        backbone_cnn = ResnetFeatureExtractor(backbone_source, backbone_path, fixed_blocks=2)
        img_enc = EncoderImageFull(backbone_cnn, img_dim, embed_size, precomp_enc_type, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.precomp_enc_type = precomp_enc_type
        if precomp_enc_type == 'basic':
            self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)
        self.linear1 = nn.Linear(embed_size, embed_size)
        self.linear2 = nn.Linear(embed_size, embed_size)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, image_lengths,gpool2=False):
        if gpool2:
            img_emb = images
            features_in = self.linear2(images)
            attn = nn.Softmax(dim=1)(features_in - torch.max(features_in, dim=1)[0].unsqueeze(1))
            feature_img = torch.sum(attn * img_emb, dim=1)

            if not self.no_imgnorm:
                feature_img = l2norm(feature_img, dim=-1)

            return feature_img
        """Extract image feature vectors."""
        features = self.fc(images)
        if self.precomp_enc_type == 'basic':
            # When using pre-extracted region features, add an extra MLP for embedding transformation
            features = self.mlp(images) + features

        img_emb= features
        features_in = self.linear1(features)
        attn = nn.Softmax(dim=1)(features_in-torch.max(features_in,dim=1)[0].unsqueeze(1))
        feature_img = torch.sum(attn * img_emb,dim=1)

        if not self.no_imgnorm:
            feature_img = l2norm(feature_img, dim=-1)

        return feature_img,features_in


class EncoderImageFull(nn.Module):
    def __init__(self, backbone_cnn, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageFull, self).__init__()
        self.backbone = backbone_cnn
        self.image_encoder = EncoderImageAggr(img_dim, embed_size, precomp_enc_type, no_imgnorm)
        self.backbone_freezed = False

    def forward(self, images):
        """Extract image feature vectors."""
        base_features = self.backbone(images)

        if self.training:
            # Size Augmentation during training, randomly drop grids
            base_length = base_features.size(1)
            features = []
            feat_lengths = []
            rand_list_1 = np.random.rand(base_features.size(0), base_features.size(1))
            rand_list_2 = np.random.rand(base_features.size(0))
            for i in range(base_features.size(0)):
                if rand_list_2[i] > 0.2:
                    feat_i = base_features[i][np.where(rand_list_1[i] > 0.20 * rand_list_2[i])]
                    len_i = len(feat_i)
                    pads_i = torch.zeros(base_length - len_i, base_features.size(-1)).to(base_features.device)
                    feat_i = torch.cat([feat_i, pads_i], dim=0)
                else:
                    feat_i = base_features[i]
                    len_i = base_length
                feat_lengths.append(len_i)
                features.append(feat_i)
            base_features = torch.stack(features, dim=0)
            base_features = base_features[:, :max(feat_lengths), :]
            feat_lengths = torch.tensor(feat_lengths).to(base_features.device)
        else:
            feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device)
            feat_lengths[:] = base_features.size(1)

        features = self.image_encoder(base_features, feat_lengths)

        return features

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info('Backbone freezed.')

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.backbone.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True
        self.backbone.set_fixed_blocks(fixed_blocks)
        self.backbone.unfreeze_base()
        logger.info('Backbone unfreezed, fixed blocks {}'.format(self.backbone.get_fixed_blocks()))


# Language Model with BiGRU
class EncoderText(nn.Module):
    def __init__(self, vocab_size, embed_size, word_dim, num_layers,opt, use_bi_gru=True, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        self.linear1 = nn.Linear(embed_size, embed_size)
        self.linear2 = nn.Linear(embed_size, embed_size)
        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        vocab = json.load(open(os.path.join(opt.vocab_path, opt.data_name + '_precomp_vocab.json'), 'rb'))
        word2idx = vocab['word2idx']
        self.init_weights(opt, word2idx)

    def init_weights(self, opt, word2idx):
        wemb = torchtext.vocab.GloVe(cache=os.path.join(opt.vocab_path, '.vector_cache'))
        assert wemb.vectors.shape[1] == opt.word_dim

        missing_words = []
        for word, idx in word2idx.items():
            if word not in wemb.stoi:
                word = word.replace('-', '').replace('.', '').replace("'", '')
                if '/' in word:
                    word = word.split('/')[0]
            if word in wemb.stoi:
                self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
            else:
                missing_words.append(word)
        print('Words: {}/{} found in vocabulary; {} words missing'.format(
            len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x=None, lengths=None,cap_emb=None,gpool2=False):
        if gpool2:
            max_len = int(lengths.max())
            mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
            mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)

            cap_emb = cap_emb[:, :int(lengths.max()), :]
            cap_external = self.linear2(cap_emb)
            cap_external = cap_external.masked_fill(mask == 0, -10000)
            # attn
            attn = nn.Softmax(dim=1)(cap_external - torch.max(cap_external, dim=1)[0].unsqueeze(1))
            attn = attn.masked_fill(mask == 0, 0)
            feature_cap = torch.sum(attn * cap_emb, dim=1)

            # normalization in the joint embedding space
            if not self.no_txtnorm:
                feature_cap = l2norm(feature_cap, dim=-1)

            return feature_cap
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x_emb = self.embed(x)

        self.rnn.flatten_parameters()
        packed = pack_padded_sequence(x_emb, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
        cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] + cap_emb[:, :, cap_emb.size(2) // 2:]) / 2
        cap_embs=cap_emb
        max_len = int(lengths.max())
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)

        cap_emb = cap_emb[:, :int(lengths.max()), :]
        cap_external = self.linear1(cap_emb)
        cap_external = cap_external.masked_fill(mask == 0, -10000)
        # attn
        attn = nn.Softmax(dim=1)(cap_external - torch.max(cap_external, dim=1)[0].unsqueeze(1))
        attn = attn.masked_fill(mask == 0, 0)
        feature_cap = torch.sum(attn * cap_emb, dim=1)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            feature_cap = l2norm(feature_cap, dim=-1)

        return feature_cap,cap_embs