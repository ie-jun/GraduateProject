from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        # below version is possible too.
        # x = torch.einsum('vw,ncwl->ncvl',(A,x))
        return x.contiguous()

class nconv2(nn.Module):
    # It is used to compute the dynamic graph's GC layer
    def __init__(self):
        super(nconv2,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,nvw->ncvl',(x,A))
        # below version is possible too.
        # x = torch.einsum('nvw,ncwl->ncvl',(A,x))
        return x.contiguous()
class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nvwl->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class prop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.reshape(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
        ho = self.mlp(h)
        return ho


class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.nconv2 = nconv2()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


    def forward(self,x,adj):
        if len(adj.shape) == 3: # for dynamic graph
            adj = adj + torch.eye(adj.size(1)).to(x.device)  # (bs , n , n ) shape
            d = adj.sum(2)
            h = x
            out = [h]

            a = []
            for i in range(d.shape[0]):
                a_i = adj[i] / d[i].reshape(-1, 1)
                a.append(a_i)
            a = torch.stack(a, dim=0) # (bs , n , n ) shape

            for i in range(self.gdep):
                h = self.alpha * x + (1 - self.alpha) * self.nconv2(h, a)
                out.append(h)
            ho = torch.cat(out, dim=1)
            ho = self.mlp(ho)
            return ho

        else:
            adj = adj + torch.eye(adj.size(0)).to(x.device) # (n , n ) shape
            d = adj.sum(1)
            h = x
            out = [h]
            a = adj / d.reshape(-1, 1)
            for i in range(self.gdep):
                h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
                out.append(h)
            ho = torch.cat(out,dim=1)
            ho = self.mlp(ho)
            return ho

class dy_mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep+1)*c_in,c_out)
        self.mlp2 = linear((gdep+1)*c_in,c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in,c_in)
        self.lin2 = linear(c_in,c_in)


    def forward(self,x):
        #adj = adj + torch.eye(adj.size(0)).to(x.device)
        #d = adj.sum(1)
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2,1),x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2,1), dim=2)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,adj0)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho1 = self.mlp1(ho)


        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1+ho2



class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        self.tconv = nn.Conv2d(cin,cout,(1,7),dilation=(1,dilation_factor))

    def forward(self,input):
        x = self.tconv(input)
        return x

class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj

class new_graph_constructor(nn.Module):
    def __init__(self, nnodes, predefined_A, in_dim,hidden_channels, seq_length, layer_depth, gcn_depth, dropout,propalpha,new_graph_only_TC,
                                                                            dilation_exponential=1,layer_norm_affline=True):
        super(new_graph_constructor, self).__init__()
        self.nnodes = nnodes
        self.predefined_A = predefined_A
        self.in_dim = in_dim
        self.hidden_channels = hidden_channels
        self.seq_length = seq_length
        self.layer_depth = layer_depth
        self.gcn_depth = gcn_depth
        self.dropout = dropout
        self.propalpha = propalpha
        self.dilation_exponential = dilation_exponential
        self.layer_norm_affline = layer_norm_affline
        self.new_graph_only_TC = new_graph_only_TC

        # About Layers
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=hidden_channels,
                                    kernel_size=(1, 1))

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()

        self.norm = nn.ModuleList()

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layer_depth-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layer_depth*(kernel_size-1) + 1

        rf_size_0 = 1
        new_dilation = 1
        for j in range(1, layer_depth + 1):
            if dilation_exponential > 1:
                rf_size_j = int(
                    rf_size_0 + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
            else:
                rf_size_j = rf_size_0 + j * (kernel_size - 1)

            self.filter_convs.append(dilated_inception(hidden_channels, hidden_channels, dilation_factor=new_dilation))
            self.gate_convs.append(dilated_inception(hidden_channels, hidden_channels, dilation_factor=new_dilation))

            ''' self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                 out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
            
            if self.seq_length > self.receptive_field:
                self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, self.seq_length - rf_size_j + 1)))
            else:
                self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, self.receptive_field - rf_size_j + 1)))

            '''
            self.gconv1.append(mixprop(hidden_channels, hidden_channels, gcn_depth, dropout, propalpha))
            self.gconv2.append(mixprop(hidden_channels, hidden_channels, gcn_depth, dropout, propalpha))

            # This module is used if residual is used.
            if self.seq_length > self.receptive_field:
                self.norm.append(LayerNorm((hidden_channels, nnodes, self.seq_length - rf_size_j + 1),
                                           elementwise_affine=layer_norm_affline))
            else:
                self.norm.append(LayerNorm((hidden_channels, nnodes, self.receptive_field - rf_size_j + 1),
                                           elementwise_affine=layer_norm_affline))

            new_dilation *= dilation_exponential

        if self.seq_length>self.receptive_field:
            self.TC_summarize_conv = nn.Conv2d(hidden_channels, hidden_channels,(1,self.seq_length-self.receptive_field+1))
            self.GC_summarize_conv = nn.Conv2d(hidden_channels, hidden_channels, (1, self.seq_length))
        else:
            self.TC_summarize_conv = nn.Conv2d(hidden_channels, hidden_channels,(1,1))
            self.GC_summarize_conv = nn.Conv2d(hidden_channels, hidden_channels, (1, self.receptive_field))

        # TC만 사용하는 version
        if self.new_graph_only_TC:
            self.out_conv = nn.Conv2d(hidden_channels, self.nnodes, (1, 1))
        else:
            self.out_conv = nn.Conv2d(hidden_channels, self.nnodes, (1, 2))




    def forward(self, x_batch , predefined_A):
        seq_len = x_batch.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            x_batch = nn.functional.pad(x_batch,(self.receptive_field-self.seq_length,0,0,0))

        x = self.start_conv(x_batch)

        tc_input = x
        gc_input = x

        # Getting Temporal features
        for i in range(self.layer_depth):
            filter = torch.tanh(self.filter_convs[i](tc_input))
            gate = torch.sigmoid(self.gate_convs[i](tc_input))

            tc_input = filter * gate
            tc_input = F.dropout(tc_input, self.dropout, training=self.training)
        tc_output = self.TC_summarize_conv(tc_input)

        if self.new_graph_only_TC:
            output_data = self.out_conv(tc_output)
            adj = torch.sigmoid(output_data)

            return adj.squeeze()

        else:
            # Getting Spatial features
            for i in range(self.layer_depth):
                gc_input = self.gconv1[i](gc_input, predefined_A)+self.gconv2[i](gc_input, predefined_A.transpose(1,0))
            gc_output = self.GC_summarize_conv(gc_input)

            concated_data = torch.cat((tc_output,gc_output),dim=-1)
            concated_data = self.out_conv(concated_data)

            adj = torch.sigmoid(concated_data)

            return adj.squeeze()


class graph_global(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

    def forward(self, idx):
        return F.relu(self.A)


class graph_undirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj



class graph_directed(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx.long(),:], self.bias[:,idx.long(),:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
