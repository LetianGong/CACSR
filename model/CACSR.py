from asyncio import base_tasks
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from utils import DotDict
from model.utils import *
import torch.nn.functional as F
from torch import nn
import random

class CACSR_ModelConfig(DotDict):
    '''
    configuration of the CACSR
    '''

    def __init__(self, loc_size=None, tim_size=None, uid_size=None, tim_emb_size=None, loc_emb_size=None,
                 hidden_size=None, user_emb_size=None, device=None,
                 loc_noise_mean=None, loc_noise_sigma=None, tim_noise_mean=None, tim_noise_sigma=None,
                 user_noise_mean=None, user_noise_sigma=None, tau=None,
                 pos_eps=None, neg_eps=None, dropout_rate_1=None, dropout_rate_2=None, rnn_type='BiLSTM',
                 num_layers=3, downstream='POI_RECOMMENDATION'):
        super().__init__()
        self.loc_size = loc_size  # 
        self.uid_size = uid_size  # 
        self.tim_size = tim_size  # 
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.user_emb_size = user_emb_size
        self.hidden_size = hidden_size  # RNN hidden_size
        self.device = device
        self.rnn_type = rnn_type
        self.num_layers = num_layers

        self.loc_noise_mean = loc_noise_mean
        self.loc_noise_sigma = loc_noise_sigma
        self.tim_noise_mean = tim_noise_mean
        self.tim_noise_sigma = tim_noise_sigma
        self.user_noise_mean = user_noise_mean
        self.user_noise_sigma = user_noise_sigma
        self.tau = tau
        self.pos_eps = pos_eps
        self.neg_eps = neg_eps
        self.dropout_rate_1 = dropout_rate_1
        self.dropout_rate_2 = dropout_rate_2
        self.downstream = downstream


class CACSR(nn.Module):
    def __init__(self, config):
        super(CACSR, self).__init__()
        # initialize parameters
        # print(config['dataset_class'])
        self.loc_size = config['loc_size']
        self.loc_emb_size = config['loc_emb_size']
        self.tim_size = config['tim_size']
        self.tim_emb_size = config['tim_emb_size']
        self.user_size = config['uid_size']
        self.user_emb_size = config['user_emb_size']
        self.hidden_size = config['hidden_size']
        # add by Tianyi (rnn_type & num_layers)
        self.rnn_type = config['rnn_type']
        self.num_layers = config['num_layers']
        self.device = config['device']
        self.downstream = config['downstream']

        if self.rnn_type == 'BiLSTM':
            self.bi = 2
        else:
            self.bi = 1

        ##############################################
        self.loc_noise_mean = config['loc_noise_mean']
        self.loc_noise_sigma = config['loc_noise_sigma']
        self.tim_noise_mean = config['tim_noise_mean']
        self.tim_noise_sigma = config['tim_noise_sigma']
        self.user_noise_mean = config['user_noise_mean']
        self.user_noise_sigma = config['user_noise_sigma']

        self.tau = config['tau']
        self.pos_eps = config['pos_eps']
        self.neg_eps = config['neg_eps']
        self.dropout_rate_1 = config['dropout_rate_1']
        self.dropout_rate_2 = config['dropout_rate_2']

        self.dropout_1 = nn.Dropout(self.dropout_rate_1)
        self.dropout_2 = nn.Dropout(self.dropout_rate_2)
        ################################################

        # Embedding layer
        self.emb_loc = nn.Embedding(num_embeddings=self.loc_size, embedding_dim=self.loc_emb_size)
        self.emb_tim = nn.Embedding(num_embeddings=self.tim_size, embedding_dim=self.tim_emb_size)
        self.emb_user = nn.Embedding(num_embeddings=self.user_size, embedding_dim=self.user_emb_size)

        # lstm layer
        # modified by Tianyi (3 kinds of rnn)
        if self.rnn_type == 'GRU':
            self.lstm = nn.GRU(self.loc_emb_size + self.tim_emb_size, self.hidden_size, num_layers=self.num_layers,
                               batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.lstm = nn.LSTM(self.loc_emb_size + self.tim_emb_size, self.hidden_size, num_layers=self.num_layers,
                                batch_first=True)
        elif self.rnn_type == 'BiLSTM':
            self.lstm = nn.LSTM(self.loc_emb_size + self.tim_emb_size, self.hidden_size, num_layers=self.num_layers,
                                batch_first=True, bidirectional=True)
        else:
            raise ValueError("rnn_type should be ['GRU', 'LSTM', 'BiLSTM']")

        if self.downstream == 'TUL':
            self.dense = nn.Linear(in_features=self.hidden_size * self.bi, out_features=self.user_size)
            self.projection = nn.Sequential(nn.Linear(self.hidden_size * self.bi, self.hidden_size * self.bi), nn.ReLU())
        elif self.downstream == 'POI_RECOMMENDATION':
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size * self.bi + self.user_emb_size, self.hidden_size * self.bi + self.user_emb_size),
                nn.ReLU())
            # dense layer
            self.dense = nn.Linear(in_features=self.hidden_size * self.bi + self.user_emb_size, out_features=self.loc_size)
        else:
            raise ValueError('downstream should in [TUL, POI_RECOMMENDATION]!')
        # self.dense_adv = nn.Linear(in_features=self.hidden_size, out_features=self.loc_size)
        # init weight
        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)

    def generate_adv(self, Anchor_hiddens, lm_labels):
        Anchor_hiddens = Anchor_hiddens.detach()
        lm_labels = lm_labels.detach()
        Anchor_hiddens.requires_grad_(True)

        Anchor_logits = self.dense(Anchor_hiddens)

        Anchor_logits = F.log_softmax(Anchor_logits, -1)

        criterion = nn.CrossEntropyLoss()
        loss_adv = criterion(Anchor_logits,
                             lm_labels).requires_grad_()

        loss_adv.backward()
        dec_grad = Anchor_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)

        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturbed_Anc = Anchor_hiddens + self.neg_eps * dec_grad.detach()
        perturbed_Anc = perturbed_Anc  # [b,t,d]

        self.zero_grad()

        return perturbed_Anc

    def generate_cont_adv(self, STNPos_hiddens,
                          Anchor_hiddens, pred,
                          tau, eps):
        STNPos_hiddens = STNPos_hiddens.detach()
        Anchor_hiddens = Anchor_hiddens.detach()
        Anchor_logits = pred.detach()
        STNPos_hiddens.requires_grad = True
        Anchor_logits.requires_grad = True
        Anchor_hiddens.requires_grad = True


        avg_STNPos = self.projection(STNPos_hiddens)
        avg_Anchor = self.projection(Anchor_hiddens)

        cos = nn.CosineSimilarity(dim=-1)
        logits = cos(avg_STNPos.unsqueeze(1), avg_Anchor.unsqueeze(0)) / tau

        cont_crit = nn.CrossEntropyLoss()
        labels = torch.arange(avg_STNPos.size(0),
                              device=STNPos_hiddens.device)
        loss_cont_adv = cont_crit(logits, labels)
        loss_cont_adv.backward()

        dec_grad = Anchor_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)
        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_Anchor_hidden = Anchor_hiddens + eps * dec_grad
        perturb_Anchor_hidden = perturb_Anchor_hidden.detach()
        perturb_Anchor_hidden.requires_grad = True
        perturb_logits = self.dense(perturb_Anchor_hidden)
        # perturb_logits = nn.LogSoftmax(dim=1)(perturb_logits)

        true_probs = F.softmax(Anchor_logits, -1)
        # true_probs = true_probs * dec_mask.unsqueeze(-1).float()

        perturb_log_probs = F.log_softmax(perturb_logits, -1)

        kl_crit = nn.KLDivLoss(reduction="sum")
        vocab_size = Anchor_logits.size(-1)

        kl = kl_crit(perturb_log_probs.view(-1, vocab_size),
                     true_probs.view(-1, vocab_size))
        kl = kl / torch.tensor(true_probs.shape[0]).float()
        kl.backward()

        kl_grad = perturb_Anchor_hidden.grad.detach()

        l2_norm = torch.norm(kl_grad, dim=-1)

        kl_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_Anchor_hidden = perturb_Anchor_hidden - eps * kl_grad

        return perturb_Anchor_hidden

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length

        return avg_hidden

    def forward(self, batch, mode='test', adv=1, downstream='POI_RECOMMENDATION'):
        loc = batch.X_all_loc
        tim = batch.X_all_tim
        user = batch.X_users
        cur_len = batch.target_lengths  
        all_len = batch.X_lengths  
        batch_size = batch.X_all_loc.shape[0]
        indice = torch.tensor(random.sample(range(loc.shape[0]), min(loc.shape[0], batch_size * 16)))

        if downstream == 'POI_RECOMMENDATION':
            loc_cpu = loc.cpu()
            tim_cpu = tim.cpu()
            user_cpu = user.cpu()
            del loc, tim, user
            loc = torch.repeat_interleave(loc_cpu, torch.tensor(cur_len), dim=0)
            tim = torch.repeat_interleave(tim_cpu, torch.tensor(cur_len), dim=0)
            user = torch.repeat_interleave(user_cpu, torch.tensor(cur_len), dim=0)
            all_len_tsr = torch.repeat_interleave(torch.tensor(all_len), torch.tensor(cur_len), dim=0)
            cur_len_tsr = torch.repeat_interleave(torch.tensor(cur_len), torch.tensor(cur_len), dim=0)

            cnt = 0
            for i in range(batch_size):
                for j in range(cur_len[i]):
                    loc[cnt, all_len[i] - cur_len[i] + j + 1:] = 0
                    tim[cnt, all_len[i] - cur_len[i] + j + 1:] = 0
                    all_len_tsr[cnt] = all_len[i] - cur_len[i] + j + 1
                    cur_len_tsr[cnt] = 1
                    cnt += 1
            assert loc.shape[0] == tim.shape[0]

            loc = torch.index_select(loc, dim=0, index=indice)
            tim = torch.index_select(tim, dim=0, index=indice)
            user = torch.index_select(user, dim=0, index=indice)
            all_len_tsr = torch.index_select(all_len_tsr, dim=0, index=indice)
            cur_len_tsr = torch.index_select(cur_len_tsr, dim=0, index=indice)

            batch_size = loc.shape[0]

            all_len = all_len_tsr.numpy().tolist()
            cur_len = cur_len_tsr.numpy().tolist()
            loc_emb = self.emb_loc(loc.to(self.device))
            tim_emb = self.emb_tim(tim.to(self.device))
            user_emb = self.emb_user(user.to(self.device))
            indice = indice.to(self.device)
        elif downstream == 'TUL':
            loc_emb = self.emb_loc(loc)
            tim_emb = self.emb_tim(tim)
            user_emb = self.emb_user(user)

        if mode == 'train' and adv == 1:
            loc_noise = torch.normal(self.loc_noise_mean, self.loc_noise_sigma, loc_emb.shape).to(loc_emb.device)
            tim_noise = torch.normal(self.tim_noise_mean, self.tim_noise_sigma, tim_emb.shape).to(loc_emb.device)
            user_noise = torch.normal(self.user_noise_mean, self.user_noise_sigma, user_emb.shape).to(loc_emb.device)

            loc_emb_STNPos = loc_emb + loc_noise
            tim_emb_STNPos = tim_emb + tim_noise
            user_emb_STNPos = user_emb + user_noise
            x_STNPos = torch.cat([loc_emb_STNPos, tim_emb_STNPos], dim=2).permute(1, 0, 2)  # batch_first=False
            pack_x_STNPos = pack_padded_sequence(x_STNPos, lengths=all_len, enforce_sorted=False)
            # modified by Tianyi
            if self.rnn_type == 'GRU':
                lstm_out_STNPos, h_n_STNPos = self.lstm(pack_x_STNPos)  # max_len*batch*hidden_size
            elif self.rnn_type == 'LSTM':
                lstm_out_STNPos, (h_n_STNPos, c_n_STNPos) = self.lstm(pack_x_STNPos)  # max_len*batch*hidden_size
            elif self.rnn_type == 'BiLSTM':
                lstm_out_STNPos, (h_n_STNPos, c_n_STNPos) = self.lstm(pack_x_STNPos)  # max_len*batch*hidden_size
            else:
                raise ValueError('rnn_type is not in [GRU, LSTM, BiLSTM]!')

            lstm_out_STNPos, out_len_STNPos = pad_packed_sequence(lstm_out_STNPos, batch_first=True)

            if downstream == 'POI_RECOMMENDATION':
                final_out_STNPos = lstm_out_STNPos[0, (all_len[0] - cur_len[0]): all_len[0], :]
                all_user_emb_STNPos = user_emb_STNPos[0].unsqueeze(dim=0).repeat(cur_len[0], 1) 
                for i in range(1, batch_size):  
                    final_out_STNPos = torch.cat(
                        [final_out_STNPos, lstm_out_STNPos[i, (all_len[i] - cur_len[i]): all_len[i], :]], dim=0)
                    all_user_emb_STNPos = torch.cat(
                        [all_user_emb_STNPos, user_emb_STNPos[i].unsqueeze(dim=0).repeat(cur_len[i], 1)], dim=0)
                final_out_STNPos = torch.cat([final_out_STNPos, all_user_emb_STNPos], 1)
            elif downstream == 'TUL':
                final_out_STNPos = lstm_out_STNPos[0, (all_len[0] - 1): all_len[0], :]
                for i in range(1, batch_size): 
                    final_out_STNPos = torch.cat(
                        [final_out_STNPos, lstm_out_STNPos[i, (all_len[i] - 1): all_len[i], :]], dim=0)
            else:
                raise ValueError('downstream is not in [POI_RECOMMENDATION, TUL]')

        # concatenate and permute
        x = torch.cat([loc_emb, tim_emb], dim=2).permute(1, 0, 2)  # batch_first=False
        # pack
        pack_x = pack_padded_sequence(x, lengths=all_len, enforce_sorted=False)

        # modified by Tianyi
        if self.rnn_type == 'GRU':
            lstm_out, h_n = self.lstm(pack_x)  # max_len*batch*hidden_size
        elif self.rnn_type == 'LSTM':
            lstm_out, (h_n, c_n) = self.lstm(pack_x)  # max_len*batch*hidden_size
        elif self.rnn_type == 'BiLSTM':
            lstm_out, (h_n, c_n) = self.lstm(pack_x)  # max_len*batch*hidden_size

        # unpack
        lstm_out, out_len = pad_packed_sequence(lstm_out, batch_first=True)

        # out_len即all_len batch*max_len*hidden_size
        # concatenate
        if downstream == 'POI_RECOMMENDATION':
            final_out = lstm_out[0, (all_len[0] - cur_len[0]): all_len[0], :]
            all_user_emb = user_emb[0].unsqueeze(dim=0).repeat(cur_len[0], 1)  
            for i in range(1, batch_size):  
                final_out = torch.cat([final_out, lstm_out[i, (all_len[i] - cur_len[i]): all_len[i], :]], dim=0)
                all_user_emb = torch.cat([all_user_emb, user_emb[i].unsqueeze(dim=0).repeat(cur_len[i], 1)], dim=0)
            final_out = torch.cat([final_out, all_user_emb], 1)
        elif downstream == 'TUL':
            final_out = lstm_out[0, (all_len[0] - 1): all_len[0], :]
            for i in range(1, batch_size):  
                final_out = torch.cat([final_out, lstm_out[i, (all_len[i] - 1): all_len[i], :]], dim=0)
        else:
            raise ValueError('downstream is not in [POI_RECOMMENDATION, TUL]')
        dense = self.dense(final_out)  # Batch * loc_size

        # dense_STNPos = self.dense(final_out_STNPos)  # Batch * loc_size
        # print(dense_STNPos)
        pred = nn.LogSoftmax(dim=1)(dense)  # result 

        ####################   adv  start    #####################
        if mode == 'train' and adv == 1:
            final_out_STNPos = self.dropout_1(final_out_STNPos)
            final_out = self.dropout_2(final_out)

            # proj_enc_h = self.projection(hidden_states)
            # proj_dec_h = self.projection(sequence_output)

            avg_STNPos = self.projection(final_out_STNPos)
            avg_Anchor = self.projection(final_out)

            cos = nn.CosineSimilarity(dim=-1)
            cont_crit = nn.CrossEntropyLoss()
            sim_matrix = cos(avg_STNPos.unsqueeze(1),
                             avg_Anchor.unsqueeze(0))
            if downstream == 'POI_RECOMMENDATION':
                # adv_imposter = self.generate_adv(final_out, batch.Y_location)  # [n,b,t,d] or [b,t,d]
                adv_imposter = self.generate_adv(final_out, torch.index_select(torch.tensor(batch.Y_location).to(self.device), dim=0, index=indice).to(self.device))  # [n,b,t,d] or [b,t,d]

            elif downstream == 'TUL':
                adv_imposter = self.generate_adv(final_out, batch.X_users)  # [n,b,t,d] or [b,t,d]
            else:
                raise ValueError('downstream is not in [POI_RECOMMENDATION, TUL]')

            batch_size = final_out.size(0)

            avg_adv_imposter = self.projection(adv_imposter)
            # avg_pert = self.avg_pool(proj_pert_dec_h,
            #                          decoder_attention_mask)

            adv_sim = cos(avg_STNPos, avg_adv_imposter).unsqueeze(1)  # [b,1]

            adv_disTarget = self.generate_cont_adv(final_out_STNPos,  # todo
                                                   final_out, dense,
                                                   self.tau, self.pos_eps)
            avg_adv_disTarget = self.projection(adv_disTarget)

            pos_sim = cos(avg_STNPos, avg_adv_disTarget).unsqueeze(-1)  # [b,1]
            logits = torch.cat([sim_matrix, adv_sim], 1) / self.tau

            identity = torch.eye(batch_size, device=final_out.device)
            pos_sim = identity * pos_sim
            neg_sim = sim_matrix.masked_fill(identity == 1, 0)
            new_sim_matrix = pos_sim + neg_sim
            new_logits = torch.cat([new_sim_matrix, adv_sim], 1)

            labels = torch.arange(batch_size,
                                  device=final_out.device)

            cont_loss = cont_crit(logits, labels)
            new_cont_loss = cont_crit(new_logits, labels)

            cont_loss = 0.5 * (cont_loss + new_cont_loss)
        ####################   adv  end     #####################
        criterion = nn.NLLLoss().to(self.device)  #

        if downstream == 'POI_RECOMMENDATION':
            s_loss_score = criterion(pred, torch.index_select(torch.tensor(batch.Y_location).to(self.device), dim=0, index=indice)).requires_grad_(True)
            _, top_k_pred = torch.topk(pred, k=self.loc_size)  # (batch, K)=(batch, num_class) 
        elif downstream == 'TUL':
            s_loss_score = criterion(pred, batch.X_users).requires_grad_(True)
            _, top_k_pred = torch.topk(pred, k=self.user_size)  # (batch, K)=(batch, num_class)  
        else:
            raise ValueError('downstream is not in [POI_RECOMMENDATION, TUL]')

        if mode == 'train' and adv == 1:
            return s_loss_score, cont_loss, top_k_pred, indice
        else:
            return s_loss_score, top_k_pred, indice
