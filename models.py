import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ConvAttentionNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, k1, k2, w1, w2, w3):
        super(ConvAttentionNetwork, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.k1 = 8
        self.k2 = 8
        self.w1 = 24
        self.w2 = 29
        self.w3 = 19

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = nn.GRU(self.embedding_dim, self.k2)
        self.attn_feat = AttentionFeatures(self.embedding, self.embedding_dim, self.k1, self.w1, self.k2, self.w2, self.w3)
        self.conv_attn_weights = AttentionWeights(self.k2, self.w3)
        self.bias = nn.Parameter(torch.ones(vocab_size)) #this should be initialized to: "the log of the empirical freq. of each target token in the training set"

    def forward(self, b, n_prev):

        n_padding = b.shape[1] - (b.shape[1] - (self.w1-1) - (self.w2-1) - (self.w3-1))
        
        if torch.cuda.is_available():
            padding = Variable(torch.zeros(b.shape[0], n_padding).long()).cuda()
        else:
            padding = Variable(torch.zeros(b.shape[0], n_padding).long()) # [bsz, n_padding]
        
        _b = torch.cat((b, padding), dim=1)

        embedded_b = self.embedding(_b)
        embedded_prev = self.embedding(n_prev) 
        _, ht = self.rnn(embedded_prev.permute(1, 0, 2))
        Lfeat = self.attn_feat(embedded_b, ht)
        a = self.conv_attn_weights(Lfeat)

        #apply the attention
        assert b.shape == a.shape
        assert embedded_b[:,:b.shape[1],:].shape[1] == a.shape[1]
        
        #TODO: i don't know if this is right, why would you just sum the attention'd vectors? That doesn't seem right.

        _nhat = a.unsqueeze(2) * embedded_b[:,:b.shape[1],:] #need to trim as we embedded w/ extra padding for the convs

        nhat = _nhat.sum(dim=1)

        #nhat = [bsz, 1, D]
        #E = [bsz, vocab_size, D]
        #transpose nhat to [bsz, D, 1]
        #batch matrix multiply -> [bsz, vocab_size, 1]
        #add the bias

        E = self.embedding.weight.unsqueeze(0).expand(nhat.shape[0], -1, -1) #get embedding weights and transform into right shape
        nhatT = nhat.unsqueeze(1).permute(0, 2, 1) #add extra dim and transpose

        n = F.softmax(torch.bmm(E, nhatT).squeeze(2) + self.bias, dim=1)

        assert n.shape[1] == self.vocab_size

        return n

class CopyAttentionNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, k1, k2, w1, w2, w3):
        super(CopyAttentionNetwork, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.k1 = k1
        self.k2 = k2
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = nn.GRU(self.embedding_dim, self.k2)
        self.attn_feat = AttentionFeatures(self.embedding, self.embedding_dim, self.k1, self.w1, self.k2, self.w2, self.w3)
        self.conv_attn_weights = AttentionWeights(self.k2, self.w3)
        self.copy_attn_weights = AttentionWeights(self.k2, self.w3)
        self.lambda_attn_weights = AttentionWeights(self.k2, self.w3)
        self.bias = nn.Parameter(torch.ones(vocab_size)) #this should be initialized to: "the log of the empirical freq. of each target token in the training set"

    def forward(self, b, n_prev):

        n_padding = b.shape[1] - (b.shape[1] - (self.w1-1) - (self.w2-1) - (self.w3-1))
        padding = Variable(torch.zeros(b.shape[0], n_padding).long()) # [bsz, n_padding]
        _b = torch.cat((b, padding), dim=1)

        embedded_b = self.embedding(_b)
        embedded_prev = self.embedding(n_prev) 
        _, ht = self.rnn(embedded_prev.permute(1, 0, 2))
        Lfeat = self.attn_feat(embedded_b, ht)
        a = self.conv_attn_weights(Lfeat)
        k = self.copy_attn_weights(Lfeat)
        
        _lambda = self.lambda_attn_weights(Lfeat)
        _lambda = torch.max(F.sigmoid(_lambda))

        #apply the attention
        assert b.shape == a.shape
        assert b.shape == k.shape
        assert embedded_b[:,:b.shape[1],:].shape[1] == a.shape[1]
        
        #TODO: i don't know if this is right, why would you just sum the attention'd vectors? That doesn't seem right.

        _nhat = a.unsqueeze(2) * embedded_b[:,:b.shape[1],:] #need to trim as we embedded w/ extra padding for the convs

        nhat = _nhat.sum(dim=1)

        #nhat = [bsz, 1, D]
        #E = [bsz, vocab_size, D]
        #transpose nhat to [bsz, D, 1]
        #batch matrix multiply -> [bsz, vocab_size, 1]
        #add the bias

        E = self.embedding.weight.unsqueeze(0).expand(nhat.shape[0], -1, -1) #get embedding weights and transform into right shape
        nhatT = nhat.unsqueeze(1).permute(0, 2, 1) #add extra dim and transpose

        n = F.softmax(torch.bmm(E, nhatT).squeeze(2) + self.bias, dim=1)

        assert n.shape[1] == self.vocab_size

        return (_lambda*k) + (1-_lambda)*n #????????? this is 100% not right

class AttentionFeatures(nn.Module):
    """
    Page 3 of the paper
    attention_features (code tokens c, context ht-1)
     C <- lookupandpad(c, E)
     L1 <- ReLU(Conv1d(C, Kl1))
     L2 <- Conv1d(L1, Kl2) * ht-1
     Lfeat <- L2/||L2||2
     return Lfeat
    """
    def __init__(self, embedding, embedding_dim, k1, w1, k2, w2, w3):
        super(AttentionFeatures, self).__init__()

        #rmsprop w/ nesterov
        #dropout on all parameters
        #parametric leaky relus
        #gradient clipping
        #all parameters initialized w/ normal random noise around zero
        #b is initialized to the log of the empirical freq. of each target token in the training set

        #conv attn 
        #k1 = k2 = 8
        #w1 = 24
        #w2 = 29
        #w3 = 10
        #do = 0.5
        #D = 128

        #copy attn
        #k1 = 32
        #k2 = 16
        #w1 = 18
        #w2 = 19
        #w3 = 2
        #do = 0.4
        #D = 128

        self.w1 = w1
        self.k1 = k1

        self.w2 = w2
        self.k2 = k2

        self.w3 = w3 #use this to calculate padding

        self.conv1 = nn.Conv1d(embedding_dim, k1, w1)
        self.conv2 = nn.Conv1d(k1, k2, w2)
        self.rnn = nn.GRU(embedding_dim, k2)

    def forward(self, C, ht):

        C = C.permute(0, 2, 1) #input to conv needs n_channels as dim 1 #[bsz, emb_dim, max_len]

        L1 = F.relu(self.conv1(C))
    
        _L2 = self.conv2(L1)

        C = C.permute(2, 0, 1) #rnn wants [seq_len, batch_size, input_size]

        #want ht in the correct shape for multiplying w/ L1
        ht = ht.permute(1, 2, 0) #ht from [max_len, bsz, hsz] -> [bsz, hsz, max_len]

        assert _L2.shape[:2] == ht.shape[:2], print(f'_L2.shape: {_L2.shape}, ht.shape: {ht.shape}')

        L2 = _L2 * ht

        Lfeat = F.normalize(L2, p=2, dim=1)

        return Lfeat

class AttentionWeights(nn.Module):
    """
    Page 3 of the paper
    attention_features (attention features Lfeat, kernel K)
     return Softmax(Conv1d(Lfeat, K))
    """
    def __init__(self, k2, w3):
        super(AttentionWeights, self).__init__()

        self.conv1 = nn.Conv1d(k2, 1, w3)

    def forward(self, Lfeat):

        x = self.conv1(Lfeat)

        x = x.squeeze(1)

        return F.softmax(x, dim=1)
