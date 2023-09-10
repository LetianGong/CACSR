import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from torch.distributions import Uniform

class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def clamp_preserve_gradients(x, min, max):
    """Clamp the tensor while preserving gradients in the clamped region."""
    return x + (x.clamp(min, max) - x).detach()


def KLD(mu, logvar):
    '''
    the KL divergency of  Gaussian distribution with a standard normal distribution
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    :param mu: the mean (batch, dim)
    :param logvar: the log of variance (batch, dim)
    :return:
    '''
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]

def KLD_category(w):
    '''
    the KL divergency of category distribution with param=w and the uniform category distribution
    :param w: (batch, nClass)
    :return:
    '''
    nClass = w.shape[1]
    p = torch.ones_like(w)/nClass  # (batch, nClass)
    # print(p[0])
    return torch.sum(w * torch.log(w/p)) / w.shape[0]


class MLP2(nn.Module):
    """
    MLP with two outputs， one for mu, one for log(var)
    """
    def __init__(self, input_size, output_size,
                 dropout=.0, hidden_size=128, use_selu=True):
        super(MLP2, self).__init__()
        self.hidden_size = hidden_size
        if self.hidden_size > 0:
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc21 = nn.Linear(hidden_size, output_size)
            self.fc22 = nn.Linear(hidden_size, output_size)
            self.nonlinear_f = F.selu if use_selu else F.relu
            self.dropout = nn.Dropout(dropout)
        else:
            self.fc21 = nn.Linear(input_size, output_size)
            self.fc22 = nn.Linear(input_size, output_size)
            self.nonlinear_f = F.selu if use_selu else F.relu
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''

        :param x: (batch, dim)
        :return:
        '''
        # print('mlp x:', x[:3,:])
        # print('mpl self.fc1(x):', self.fc1(x)[:3, :])
        # print('mpl self.nonlinear_f(self.fc1(x)):', self.nonlinear_f(self.fc1(x))[:3, :])
        if self.hidden_size > 0:
            h1 = self.dropout(self.nonlinear_f(self.fc1(x)))
            return self.fc21(h1), self.fc22(h1)
        else:
            return self.fc21(x), self.fc22(x)


class MLP(nn.Module):
    """
    MLP with one output (not normalized) for multinomial distribution
    """
    def __init__(self, input_size, hidden_size=64, output_size=1, dropout=0.0, use_selu=True):
        '''

        :param input_size:
        :param hidden_size:
        :param output_size: the num of cluster
        :param dropout:
        :param use_selu:
        '''
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.nonlinear_f = F.selu if use_selu else F.leaky_relu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h1 = self.dropout(self.nonlinear_f(self.fc1(x)))
        return self.fc2(h1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def attention(query, key, value, mask=None, dropout=None):
    '''

    :param query: （B, <h>, max_length, d_k）
    :param key: （B, <h>, max_length, d_k）
    :param value: （B, <h>, max_length, d_k）
    :param mask:  (B, <1>, max_length, max_length), true/false matrix, and true means paddings
    :param dropout:
    :return: outputs:(B, <h>, max_length, d_k), att_scores:(B, <h>, max_length, max_length)
    '''
    "Compute 'Scaled Dot Product Attention'"
    # print('query:', query.shape)
    # print('key:', key.shape)
    # print('value:', value.shape)
    d_k = query.size(-1)
    # print('start 4 query:', query[-1,0])
    # print('start 4 key:', key[-1,0])
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print('start 4 scores:', scores.shape, scores[-1, 0, :, :])
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)  # true->-1e9
    # print('mask:', mask.shape, mask[-1,0,0,:])
    # print('start 5 scores:', scores.shape, scores[-1,0,:,:])
    p_attn = F.softmax(scores, dim=-1)  # 每行和为1
    # print('start 5 p_attn:', p_attn.shape, p_attn[-1, 0, :, :])
    # print('----')
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # for query, key, value, output
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, distance=None):
        '''

        :param query: (B, max_length, d_model)
        :param key: (B, max_length, d_model)
        :param value: (B, max_length, d_model)
        :param mask: (B, max_length, max_length)
        :return: (B, max_length, d_model)
        '''
        # print('start 3 MHA query:', query[0])
        # print('start 3 MHA key:', key[0])
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # mask (B, 1, max_length)->(B, 1, 1, max_length)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


def spatial_aware_attention(query, key, value, distance, mask=None, dropout=None):
    '''

    :param query: （B, h, max_length, d_k）
    :param key: （B, h, max_length, d_k）
    :param value: （B, h, max_length, d_k）
    :param distance: (B, h, max_length, max_length)
    :param mask:  (B, 1, max_length, max_length), true/false matrix, and true means paddings
    :param dropout:
    :return: outputs:(B, h, max_length, d_k), att_scores:(B, h, max_length, max_length)
    '''
    "Compute 'Scaled Dot Product Attention'"
    # print('query:', query.shape)
    # print('key:', key.shape)
    # print('value:', value.shape)
    d_k = query.size(-1)
    # print('start 4 query:', query[-1,0])
    # print('start 4 key:', key[-1,0])
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # （B, h, max_length, max_length）
    # print('start 4 scores:', scores.shape, scores[-1, 0, :, :])
    scores = scores - distance  # （B, h, max_length, max_length）
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)  # true->-1e9
    # print('mask:', mask.shape, mask[-1,0,0,:])
    # print('start 5 scores:', scores.shape, scores[-1,0,:,:])
    p_attn = F.softmax(scores, dim=-1)  # 每行和为1
    # print('start 5 p_attn:', p_attn.shape, p_attn[-1, 0, :, :])
    # print('----')
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class SpatialAwareMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(SpatialAwareMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # for query, key, value, output
        self.logwb = nn.Parameter(Uniform(0.0, 1.0).sample((self.h,)))  # (h,)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, distance=None):
        '''

        :param query: (B, max_length, d_model)
        :param key: (B, max_length, d_model)
        :param value: (B, max_length, d_model)
        :param distance: (B, max_length, max_length)
        :param mask: (B, max_length, max_length)
        :return: (B, max_length, d_model)
        '''
        # print('start 3 MHA query:', query[0])
        # print('start 3 MHA key:', key[0])
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # mask (B, 1, max_length)->(B, 1, 1, max_length)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # distance to multi-head: (B, max_length, max_length) --> (B, h, max_length, max_length)
        wb = torch.exp(self.logwb).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1,h,1,1)
        mh_distance = distance.unsqueeze(1).repeat(1, self.h, 1, 1) * wb  # (B, h, max_length, max_length)*(1,h,1,1)
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = spatial_aware_attention(query, key, value, mh_distance, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''

        :param x: (B, max_length, d_model)
        :return: (B, max_length, d_model)
        '''
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class TransformerEncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None, distance=None):
        '''

        :param x: (B, max_length, d_model)
        :param mask: (B, 1, max_length)
        :return: (B, max_length, d_model)
        '''
        # print('start 2 x:', x[0])
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask=mask, distance=distance))
        return self.sublayer[1](x, self.feed_forward)


class TransformerEncoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(TransformerEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, padding_mask=None, session_mask=None, subsequent_mask=None, distance=None):
        mask = torch.zeros(x.size(0), x.size(1), x.size(1), device=x.device).bool()  # .type(torch.uint8)  # (B, max_length, max_length)
        # torch.set_printoptions(threshold=1000000)
        if padding_mask is not None:
            padding_mask = padding_mask.repeat(1, x.size(1), 1).bool()  # (B, max_length, max_length)
            # print('in padding_mask:', padding_mask)
            mask = mask | padding_mask
        if session_mask is not None:
            # print('in session_mask:', session_mask)
            mask = mask | session_mask
        if subsequent_mask is not None:
            # print('in subsequent_mask:', subsequent_mask)
            mask = mask | subsequent_mask
        # print('in mask', mask)
        for layer in self.layers:
            x = layer(x, mask=mask, distance=distance)
        return self.norm(x)

class Hypernet(nn.Module):
    """
        Hypernetwork deals with decoder input and generates params for mu, sigma, w

    Args:
        config: Model configuration.
        hidden_sizes: Sizes of the hidden layers. [] corresponds to a linear layer.
        param_sizes: Sizes of the output parameters. [n_components, n_components, n_components] 分别指定w,mu,s的维度/components
        activation: Activation function.
    """
    def __init__(self, config, hidden_sizes=[], param_sizes=[1, 1, 1], activation=nn.Tanh()):
        super().__init__()
        self.decoder_input_size = config.decoder_input_size
        self.activation = activation

        # print("hidden_sizes:", hidden_sizes)  # []
        # print("param_sizes:", param_sizes)  # [64, 64, 64]
        # Indices for unpacking parameters
        ends = torch.cumsum(torch.tensor(param_sizes), dim=0)
        starts = torch.cat((torch.zeros(1).type_as(ends), ends[:-1]))
        self.param_slices = [slice(s.item(), e.item()) for s, e in zip(starts, ends)]
        # self.param_slices.shape =  [slice(0, 64, None), slice(64, 128, None), slice(128, 192, None)]

        self.output_size = sum(param_sizes)  
        layer_sizes = list(hidden_sizes) + [self.output_size]
        # print("Hypernet layer_sizes:", layer_sizes)  # [192]
        # Bias used in the first linear layer
        self.first_bias = nn.Parameter(torch.empty(layer_sizes[0]).uniform_(-0.1, 0.1))  
        self.first_linear = nn.Linear(self.decoder_input_size, layer_sizes[0], bias=False)

        # Remaining linear layers
        self.linear_layers = nn.ModuleList()
        for idx, size in enumerate(layer_sizes[:-1]):
            self.linear_layers.append(nn.Linear(size, layer_sizes[idx + 1]))

    def reset_parameters(self):

        self.first_bias.data.fill_(0.0)
        self.first_linear.reset_parameters()
        nn.init.orthogonal_(self.first_linear.weight)
        for layer in self.linear_layers:
            layer.reset_parameters()
            nn.init.orthogonal_(layer.weight)

    def forward(self, decoder_input):
        """Generate model parameters from the embeddings.

        Args:
            input: decoder input, shape (batch, decoder_input_size)

        Returns:
            params: Tuple of model parameters.
        """
        # Generate the output based on the input
        hidden = self.first_bias
        hidden = hidden + self.first_linear(decoder_input)
        for layer in self.linear_layers:
            hidden = layer(self.activation(hidden))

        # # Partition the output
        # if len(self.param_slices) == 1:
        #     return hidden
        # else:
        return tuple([hidden[..., s] for s in self.param_slices])
