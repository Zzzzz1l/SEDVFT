import numpy as np
import torch
from torch import nn
from models.containers import Module
from torch.nn import functional as F


class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

        self.comment = comment

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask.bool(), -np.inf)  # add .bool()
        att = torch.softmax(att, -1)
        att = self.dropout(att)   

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class ScaledDotProductWithBoxAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductWithBoxAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        #my_Linear
        self.fc_pv = nn.Linear(49,d_v)
        self.fc_att = nn.Linear(49,49)



        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.scale = nn.Parameter(torch.log(10 * torch.ones((h, 1, 1))), requires_grad=True)

        self.init_weights()

        self.comment = comment


    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.xavier_uniform_(self.fc_att.weight)

        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        
        nn.init.constant_(self.fc_att.bias, 0)

    def forward(self, queries, keys, values, grids_pos, box_relation_embed_matrix, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]


        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # SEA module
        pos_q = grids_pos.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        pos_k = grids_pos.view(b_s, nk, self.h, self.d_k).permute(0, 2, 1, 3)

        att_pos = torch.matmul(pos_k, pos_q)
        att_pos = self.fc_att(att_pos)

        att1 = torch.matmul(q, pos_q) #bias
        att2 = torch.matmul(pos_k, k)   #bias

        att = F.normalize(q, dim=-1) @ F.normalize(k, dim=-2)

        att = (att1 + att2 + att) / np.sqrt(self.d_k) #添加bias

        scale = torch.clamp(self.scale, max=torch.log(torch.tensor(1. / 0.01))).exp()

        w_g = box_relation_embed_matrix
        w_g = torch.sign(w_g) * torch.log(1 + torch.abs(w_g))

        att = (att + att_pos + w_g) * scale

        
        
        
        if attention_weights is not None:
            att = att * attention_weights

        if attention_mask is not None:
            att = att.masked_fill(attention_mask.bool(), -np.inf)  # add .bool()

        att = torch.softmax(att, -1)  ## bs * 8 * r * r
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out

class RMSNorm(nn.Module):
    def __init__(self, d_model, p=-1., eps=1e-8, bias=False, dropout=.1):

        """
            Root Mean Square Layer Normalization
        :param d_model: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d_model
        self.p = p
        self.bias = bias
        self.gated = nn.Sequential(nn.Linear(2 * d_model, 2 * d_model),
                                   nn.GLU())
        self.dropout = nn.Dropout(dropout)

        self.scale = nn.Parameter(torch.ones(d_model))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d_model))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        # G-RMSNorm
        return self.dropout(self.gated(torch.cat([x, self.scale * x_normed], -1)))


class ScaledDotProductAttention_rela(nn.Module): #看一下和ScaledDotProductAttention的区别
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention_rela, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.norm = RMSNorm(d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

        self.comment = comment

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        # GNA module
        out = self.norm(out)

        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class MultiHeadBoxAttention(Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, comment=None):
        super(MultiHeadBoxAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductWithBoxAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, grids_pos, box_relation_embed_matrix, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, grids_pos, box_relation_embed_matrix, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, grids_pos, box_relation_embed_matrix, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out


class MultiHeadAttention(Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, comment=None):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out

class MultiHeadAttention_rela(Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, comment=None):
        super(MultiHeadAttention_rela, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        if attention_module is not None:
            if attention_module_kwargs is not None:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, **attention_module_kwargs)
            else:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        else:
            self.attention = ScaledDotProductAttention_rela(d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out

class MyScaledDotProductWithBoxAttentionTwo(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MyScaledDotProductWithBoxAttentionTwo, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o1 = nn.Linear(h * d_v, d_model)
        self.fc_o2 = nn.Linear(h * d_v, d_model)
        self.fc_q1q2 = nn.Linear(d_model,h*d_k)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

        self.comment = comment
    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o1.weight)
        nn.init.xavier_uniform_(self.fc_o2.weight)
        nn.init.xavier_uniform_(self.fc_q1q2.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o1.bias, 0)
        nn.init.constant_(self.fc_o2.bias, 0)
        nn.init.constant_(self.fc_q1q2.bias,0)
    def forward(self,regions_out,grids_out,interests,attention_mask_regions, attention_mask_grids,attention_weights):
        
        b_s, nq = interests.shape[:2]
        nk1 = regions_out.shape[1]
        nk2 = grids_out.shape[1]
        q = self.fc_q(interests).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        q1 = self.fc_q1q2(regions_out).view(b_s, nk1, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, d_k, nk)
        q2 = self.fc_q1q2(grids_out).view(b_s, nk1, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, d_k, nk)
        k1 = self.fc_k(regions_out).view(b_s, nk1, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        k2 = self.fc_k(grids_out).view(b_s, nk2, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v1 = self.fc_v(regions_out).view(b_s, nk1, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        v2 = self.fc_v(grids_out).view(b_s, nk2, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att1 = torch.matmul(q + q2, k1) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        att2 = torch.matmul(q + q1, k2) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        if attention_weights is not None:
            att1 = att1 * attention_weights
            att2 = att2 * attention_weights
        if attention_mask_regions is not None:
            att1 = att1.masked_fill(attention_mask_regions.bool(), -np.inf)
        if attention_mask_grids is not None:
            att2 = att2.masked_fill(attention_mask_grids.bool(), -np.inf)
        att1 = torch.softmax(att1, -1)  ## bs * 8 * r * r
        att2 = torch.softmax(att2,-1)
        att1 = self.dropout1(att1)
        att2 = self.dropout2(att2)
        
        
        out1 = torch.matmul(att1, v1).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out1 = self.fc_o1(out1)  # (b_s, nq, d_model)
        out2 = torch.matmul(att2, v2).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out2 = self.fc_o2(out2)  # (b_s, nq, d_model)
        return out1,out2



class MyScaledDotProductWithBoxAttentionOne(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MyScaledDotProductWithBoxAttentionOne, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        
        self.fc_v = nn.Linear(d_model, h * d_v)
        
        self.fc_o1 = nn.Linear(h * d_v, d_model)
        self.fc_o2 = nn.Linear(h * d_v, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.myweight = nn.Parameter(torch.tensor(0.1))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

        self.comment = comment
    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        
        nn.init.xavier_uniform_(self.fc_o1.weight)
        nn.init.xavier_uniform_(self.fc_o2.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        
        
    def forward(self,regions_out,grids_out,interests,attention_mask_regions, attention_mask_grids,attention_weights):
        
        b_s, nq = interests.shape[:2]
        nk1 = regions_out.shape[1]
        nk2 = grids_out.shape[1]
        q = self.fc_q(interests).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k1 = self.fc_k(regions_out).view(b_s, nk1, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        k2 = self.fc_k(grids_out).view(b_s, nk2, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v1 = self.fc_v(regions_out).view(b_s, nk1, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        v2 = self.fc_v(grids_out).view(b_s, nk2, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att1 = torch.matmul(q, k1) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        att2 = torch.matmul(q, k2) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        if attention_weights is not None:
            att1 = att1 * attention_weights
            att2 = att2 * attention_weights
        if attention_mask_regions is not None:
            att1 = att1.masked_fill(attention_mask_regions.bool(), -np.inf)
        if attention_mask_grids is not None:
            att2 = att2.masked_fill(attention_mask_grids.bool(), -np.inf)
        att1 = torch.softmax(att1, -1)  ## bs * 8 * r * r
        att2 = torch.softmax(att2,-1)
        att1 = self.dropout1(att1)
        att2 = self.dropout2(att2)
        
        weight = torch.tensor(0.1)
        
        out1 = torch.matmul(att1 + weight*att2 , v1).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out1 = self.fc_o1(out1)  # (b_s, nq, d_model)
        out2 = torch.matmul(att2 + weight*att1 , v2).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out2 = self.fc_o2(out2)  # (b_s, nq, d_model)
        return out1,out2
    



class MyMultiHeadBoxAttention(Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, comment=None):
        super(MyMultiHeadBoxAttention, self).__init__()
        self.attention = MyScaledDotProductWithBoxAttentionOne(d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self,regions_out,grids_out,interests,attention_mask_regions, attention_mask_grids,attention_weights):
        

        out1,out2 = self.attention(regions_out,grids_out,interests,attention_mask_regions, attention_mask_grids,attention_weights)
        out1 = self.dropout1(out1)
        out2 = self.dropout2(out2)
        out1 = self.layer_norm1(regions_out + out1)
        out2 = self.layer_norm2(grids_out + out2)
        return out1,out2


        