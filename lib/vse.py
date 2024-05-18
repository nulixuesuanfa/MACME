"""VSE model"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_

from lib.encoders import get_image_encoder, get_text_encoder
from lib.loss import ContrastiveLoss

import logging

from sklearn.metrics.pairwise import cosine_similarity
import fasttext
from nltk.corpus import wordnet as wn
import random

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

logger = logging.getLogger(__name__)

class VSEModel(object):
    """
        The standard VSE model
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc =get_image_encoder(opt.img_dim, opt.embed_size,
                                         precomp_enc_type=opt.precomp_enc_type,
                                         backbone_source=opt.backbone_source,
                                         backbone_path=opt.backbone_path,
                                         no_imgnorm=opt.no_imgnorm)
        self.txt_enc = get_text_encoder(opt.vocab_size, opt.embed_size, opt.word_dim, opt.num_layers,opt,
                                        use_bi_gru=True, no_txtnorm=opt.no_txtnorm)
        #self.mix_enc = get_text_encoder(opt.vocab_size, opt.embed_size, opt.word_dim, opt.num_layers,
        #                                use_bi_gru=True, no_txtnorm=opt.no_txtnorm)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()

            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())

        self.params = params
        self.opt = opt

        # Set up the lr for different parts of the VSE model
        if opt.precomp_enc_type == 'basic':
            if self.opt.optim == 'adam':
                self.optimizer = torch.optim.AdamW(self.params, lr=opt.learning_rate)
            else:
                raise ValueError('Invalid optim option {}'.format(self.opt.optim))
        else:
            decay_factor = 1e-4
            if self.opt.optim == 'adam':
                self.optimizer = torch.optim.AdamW([
                    {'params': self.txt_enc.parameters(), 'lr': opt.learning_rate},
                    {'params': self.img_enc.backbone.top.parameters(),
                     'lr': opt.learning_rate * opt.backbone_lr_factor, },
                    {'params': self.img_enc.backbone.base.parameters(),
                     'lr': opt.learning_rate * opt.backbone_lr_factor, },
                    {'params': self.img_enc.image_encoder.parameters(), 'lr': opt.learning_rate},
                ], lr=opt.learning_rate, weight_decay=decay_factor)
            else:
                raise ValueError('Invalid optim option {}'.format(self.opt.optim))

        logger.info('Use {} as the optimizer, with init lr {}'.format(self.opt.optim, opt.learning_rate))

        self.Eiters = 0
        self.data_parallel = False

        self.text_embedding_model = fasttext.load_model('/media/hdd4/luz/vse_infty-bigru-class/cc.en.300.bin')
        self.nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}



    def set_max_violation(self, max_violation):
        if max_violation:
            self.criterion.max_violation_on()
        else:
            self.criterion.max_violation_off()

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0], strict=False)
        self.txt_enc.load_state_dict(state_dict[1], strict=False)


    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()


    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()


    def freeze_backbone(self):
        if 'backbone' in self.opt.precomp_enc_type:
            if isinstance(self.img_enc, nn.DataParallel):
                self.img_enc.module.freeze_backbone()
            else:
                self.img_enc.freeze_backbone()

    def unfreeze_backbone(self, fixed_blocks):
        if 'backbone' in self.opt.precomp_enc_type:
            if isinstance(self.img_enc, nn.DataParallel):
                self.img_enc.module.unfreeze_backbone(fixed_blocks)
            else:
                self.img_enc.unfreeze_backbone(fixed_blocks)

    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc)
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.data_parallel = True
        logger.info('Image encoder is data paralleled now.')

    @property
    def is_data_parallel(self):
        return self.data_parallel



    def mix(self, images, captions,classes_raw,captions_raw):
        images = images.clone()
        captions = captions.clone()
        for i in range(len(classes_raw)):
            classes_raw_i_emb=[self.text_embedding_model.get_word_vector(word) for word in classes_raw[i]]
            captions_noun_index=[j  for j in range(len(captions_raw[i])) if captions_raw[i][j] in self.nouns]
            if len(captions_noun_index)==0:
                #print("no noun in raw caption!")
                continue
            captions_raw_i_emb=[self.text_embedding_model.get_word_vector(captions_raw[i][wordidx]) for wordidx in captions_noun_index]
            sim=cosine_similarity(captions_raw_i_emb,classes_raw_i_emb)
            num_mix=0

            for j in range(sim.shape[0]):
                classes_similar_index=[k for k in range(len(sim[j])) if sim[j][k] > 0.7]
                if len(classes_similar_index)==0:
                    #print("no similar classes with this noun!",end=' ')
                    continue
                else:
                    num_mix+=1
                tem=captions[i][captions_noun_index[j]]
                captions[i][captions_noun_index[j]]=torch.mean(images[i][classes_similar_index])
                for ind in classes_similar_index:
                    prob = random.random()
                    if prob < 0.5:
                        images[i][ind]=tem
            #print(f'{num_mix} mixings with this caption')
        return images, captions

    def mix4(self, images, captions,classes_raw,captions_raw):
        images = images.clone()
        captions = captions.clone()
        for i in range(len(classes_raw)):
            classes_raw_i_emb=[self.text_embedding_model.get_word_vector(word) for word in classes_raw[i]]
            captions_noun_index=[j  for j in range(len(captions_raw[i])) if captions_raw[i][j] in self.nouns]
            if len(captions_noun_index)==0:
                #print("no noun in raw caption!")
                continue
            captions_raw_i_emb=[self.text_embedding_model.get_word_vector(captions_raw[i][wordidx]) for wordidx in captions_noun_index]
            sim=cosine_similarity(captions_raw_i_emb,classes_raw_i_emb)
            num_mix=0

            for j in range(sim.shape[0]):
                classes_similar_index=[k for k in range(len(sim[j])) if sim[j][k] > 0.7]
                classes_unsimilar_index = [k for k in range(len(sim[j])) if sim[j][k] <0.4]
                if len(classes_similar_index)==0:
                    #print("no similar classes with this noun!",end=' ')
                    continue
                else:
                    num_mix+=1
                tem=captions[i][captions_noun_index[j]]
                captions[i][captions_noun_index[j]]=torch.mean(images[i][classes_similar_index])
                images[i][classes_unsimilar_index]=torch.mean(captions[i][captions_noun_index[j]])
        return images, captions


    def mix_3(self, images, captions, classes_raw, captions_raw):
        for i in range(len(classes_raw)):
            classes_raw_i_emb = [self.text_embedding_model.get_word_vector(word) for word in classes_raw[i]]
            captions_noun_index = [j for j in range(len(captions_raw[i])) if captions_raw[i][j] in self.nouns]
            if len(captions_noun_index) == 0:
                # print("no noun in raw caption!")
                continue
            captions_raw_i_emb = [self.text_embedding_model.get_word_vector(captions_raw[i][wordidx]) for wordidx in
                                  captions_noun_index]
            sim = cosine_similarity(captions_raw_i_emb, classes_raw_i_emb)
            num_mix = 0
            for j in range(sim.shape[0]):
                k=np.argmax(sim[j])
                max_sim=sim[j][k]#max(sim[j])
                if max_sim>0.7:
                    #k = sim[j].index(max_sim)
                    tem = captions[i][captions_noun_index[j]]
                    captions[i][captions_noun_index[j]]=images[i][k]
                    images[i][k]=tem

                    num_mix += 1
                else:
                    continue
        return images, captions


    def mix_2(self,images,captions,classes_raw, captions_raw):
        for i in range(images.shape[0]):
            sim=cosine_similarity(captions[i].cpu().detach().numpy(), images[i].cpu().detach().numpy())
            for j in range(sim.shape[0]):
                k=np.argmax(sim[j])
                max_sim=sim[j][k]#max(sim[j])
                if max_sim>0.7:
                    #k = sim[j].index(max_sim)
                    tem = captions[i][j]
                    captions[i][j]=images[i][k]
                    images[i][k]=tem
                else:
                    continue
        return images, captions



    def forward_emb(self, images, captions, lengths, image_lengths=None,classes_raw=None,captions_raw=None,train=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            image_lengths = image_lengths.cuda()

        lengths = torch.Tensor(lengths).cuda()

        img_emb,img_embs = self.img_enc(images, image_lengths)
        cap_emb,cap_embs = self.txt_enc(x=captions, lengths=lengths)

        for i in range(captions.shape[0]):
            if 5922 in captions[i] and 5952 in captions[i]:
                print(captions[i])
                print(img_embs[i].shape)
                plt.figure(dpi=300,figsize=(6.4, 6.4))
                df = pd.DataFrame(data=1-cosine_similarity(img_embs[i].cpu(),cap_embs[i][1:11].cpu()), columns =['six', 'people', 'riding', 'bikes', 'on', 'a', 'trail', 'in', 'the', 'forest'])
                sns.heatmap(df, cmap='coolwarm',yticklabels=1,vmin=0.91, vmax=1.125)
                plt.xticks(rotation=45)
                plt.xlabel('Text')
                plt.ylabel('Image')
                plt.title('MACME w/o MIX')
                plt.savefig('MACMEwoMIX.jpg',dpi=300)


        if train:
            images_mix, captions_mix = self.mix(img_embs, cap_embs, classes_raw, captions_raw)
            captions_mix=self.txt_enc(lengths=lengths,cap_emb=captions_mix,gpool2=True)
            images_mix=self.img_enc(images_mix, image_lengths,gpool2=True)
            return img_emb, cap_emb, captions_mix,images_mix
        else:
            return img_emb,cap_emb

    def forward_loss(self, img_emb, cap_emb):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb)
        self.logger.update('Le', loss.data.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths,classes,lengths_c, c_raw,tokens,lamda1,lamda2,lamda3,image_lengths=None, warmup_alpha=None):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        '''
        # compute the embeddings
        _, img_embs= self.img_enc(images.cuda(), image_lengths.cuda())
        classes=torch.cat([i[:j] for (i,j) in zip(classes,lengths_c)])
        classes=classes.cuda()
        classes_embs,_ = self.txt_enc(x_emb=self.txt_enc.embed(classes.unsqueeze(1)), lengths=torch.ones(classes.shape[0]) )
        img_embs=torch.cat([i[:int(j)] for (i,j) in zip(img_embs,image_lengths)])
        '''
        img_emb, cap_emb,cap_mix_emb ,images_mix_emb= self.forward_emb(images, captions, lengths, image_lengths=image_lengths,classes_raw=c_raw,captions_raw=tokens,train=True)

        # measure accuracy and record loss
        self.optimizer.zero_grad()#self.forward_loss(img_emb, cap_mix_emb)
        loss = self.forward_loss(img_emb, cap_emb)+lamda1*self.forward_loss(cap_emb, cap_mix_emb)+lamda2*self.forward_loss(img_emb, images_mix_emb)+lamda3*self.forward_loss(images_mix_emb, cap_mix_emb)#+lamda3*self.forward_loss(img_embs, classes_embs.squeeze(1))

        if warmup_alpha is not None:
            loss = loss * warmup_alpha

        # compute gradient and update
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

