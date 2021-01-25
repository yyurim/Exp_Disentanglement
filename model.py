import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def average_conv_general(m, T=False):
    with torch.no_grad():
        if T:
            total_weight = m.weight.permute(1, 0, 2, 3)
        else:
            total_weight = m.weight
        
        mean_w = torch.mean(total_weight, dim=0)
        
        for outC_idx, w in enumerate(total_weight):
            total_weight[outC_idx]=mean_w
        
        if T:
            m.weight = nn.Parameter(total_weight.permute(1, 0, 2, 3)).cuda()
        else:
            m.weight = nn.Parameter(total_weight).cuda()

        
    return m

class CIN_2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CIN_2d, self).__init__()
        outC = kwargs.get("outC", args[0])
        style_dim = kwargs.get("style_dim", args[1])

        self.norm = nn.InstanceNorm2d(outC, affine=False)
        if style_dim > 0:
            self.sig = nn.Linear(style_dim, outC)
            self.mu = nn.Linear(style_dim, outC)
    
    def forward(self, x, style=None):
        norm_x = self.norm(x)
        if style is not None:
            s = self.sig(style)
            m = self.mu(style)

            s = s.view(s.size(0), s.size(1), 1, 1)
            m = m.view(m.size(0), m.size(1), 1, 1)
            
            norm_x = s*norm_x + m

        return norm_x

class Conv2d_GLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Conv2d_GLU, self).__init__()
        inC = kwargs.get("inC", 0)
        outC = kwargs.get("outC", 0)
        k = kwargs.get("k", 0)
        s = kwargs.get("s", 0)
        p = kwargs.get("p", 0)
        T = kwargs.get("transpose", False)
        # style_dim = kwargs.get("style_dim", 0)
    
        self.T = T

        if T:   
            self.cnn = nn.ConvTranspose2d(inC, outC, kernel_size=k, stride=s, padding=p)
            self.gate = nn.ConvTranspose2d(inC, outC, kernel_size=k, stride=s, padding=p)
            self.cnn_norm = nn.InstanceNorm2d(outC)
            self.gate_norm = nn.InstanceNorm2d(outC)
            self.sig = nn.Sigmoid()

        else:
            self.cnn = nn.Conv2d(inC, outC, kernel_size=k, stride=s, padding=p)
            self.gate = nn.Conv2d(inC, outC, kernel_size=k, stride=s, padding=p)
            self.cnn_norm = nn.InstanceNorm2d(outC)
            self.gate_norm = nn.InstanceNorm2d(outC)
            self.sig = nn.Sigmoid()
            
    def forward(self, x):
        
        h1 = self.cnn_norm(self.cnn(x))
        h2 = self.gate_norm(self.gate(x))
        out = torch.mul(h1, self.sig(h2))        
        return out

class DummyLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DummyLayer, self).__init__()
        spk_num = kwargs.get("spk_num", 0)

        # self.dummy = nn.Conv1d(spk_num, 1, kernel_size=1, stride=1)
        self.dummy = nn.Linear(spk_num,1,bias=False)
        self.act = nn.ReLU()

    def forward(self, spk_vec):
        out = self.act(self.dummy(spk_vec))
        return out

def attach_style(inputs, style):
    style = style.view(style.size(0), style.size(1), 1, 1)
    style = style.repeat(1, 1, inputs.size(2), inputs.size(3))
    inputs_bias_added = torch.cat([style, inputs], dim=1)
    return inputs_bias_added

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add(mu)

class Encoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Encoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.style_dim = kwargs.get("style_dim", 0)
        self.latent_dim = kwargs.get("latent_dim", 8)
        self.vae_type = kwargs.get("vae_type", '')
        self.weight_sharing = kwargs.get("weight_sharing", True)

        assert self.vae_type in ['VAE1', 'VAE2', 'VAE3', 'MD'], "VAE type error"

        """
        (1, 36, 128) => (5, 36, 128) => (10, 18, 64) => (10, 9, 32) => (16, 1, 32)
        (k-s)/2 = p
        """
    
        C_structure = [5, 10, 10, self.latent_dim]
        k_structure = [(3,9), (4,8), (4,8), (9,5)]
        s_structure = [(1,1), (2,2), (2,2), (9,1)]

        layer_num = len(C_structure)

        inC = 1
        
        self.convs= nn.ModuleList([])
        if self.vae_type in ['VAE3', 'MD'] and self.weight_sharing:
            self.dummy4ws = DummyLayer(spk_num=self.style_dim)

        for layer_idx in range(layer_num):
            if self.vae_type in ['VAE3', 'MD']:
                if self.weight_sharing:
                    inC += 1
                else:
                    inC += self.style_dim

            outC = C_structure[layer_idx]
            k = k_structure[layer_idx]
            s = s_structure[layer_idx]
            p = ((k[0]-s[0])//2, (k[1]-s[1])//2)

            if layer_idx == layer_num-1:
                self.conv_mu = nn.Conv2d(inC, outC, k, s, padding=p)
                self.conv_logvar = nn.Conv2d(inC, outC, k, s, padding=p)
            else:
                self.convs.append(
                    Conv2d_GLU(inC=inC, outC=outC, k=k, s=s, p=p)
                )
                inC = outC
                
    def forward(self, x, one_hot=None):
        h = x
        if self.vae_type in ['VAE3', 'MD']:
            if self.weight_sharing:
                ws = self.dummy4ws(one_hot)
                h = attach_style(h,ws)
            else:
                h = attach_style(h,one_hot)

        for i, conv in enumerate(self.convs):    
            h = conv(h)
            if self.vae_type in ['VAE3', 'MD']:
                if self.weight_sharing:
                    ws = self.dummy4ws(one_hot)
                    h = attach_style(h,ws)
                else:
                    h = attach_style(h,one_hot)
            
        h_mu = self.conv_mu(h)
        h_logvar = self.conv_logvar(h)
        o = reparameterize(h_mu, h_logvar)
     
        return h_mu, h_logvar, o


class Decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.style_dim = kwargs.get("style_dim", 0)
        self.latent_dim = kwargs.get("latent_dim", 8)
        self.vae_type = kwargs.get("vae_type", '')
        self.weight_sharing = kwargs.get("weight_sharing", True)

        assert self.vae_type in ['VAE1', 'VAE2', 'VAE3', 'MD'], "VAE type error"

        """
        (8, 1, 32) => (10, 9, 32) => (10, 18, 64) => (5, 36, 128) => (1, 36, 128)
        (k-s)/2 = p
        """
        C_structure = [10, 10, 5, 1]
        k_structure = [(9,5), (4,8), (4,8), (3,9)]
        s_structure = [(9,1), (2,2), (2,2), (1,1)]

        layer_num = len(C_structure)

        inC = self.latent_dim
        self.convs= nn.ModuleList([])
        if self.vae_type in ['VAE3', 'MD'] and self.weight_sharing:
            self.dummy4ws = DummyLayer(spk_num=self.style_dim)

        if self.vae_type in ['VAE1', 'VAE2', 'VAE3']:
            inC += self.style_dim

        for layer_idx in range(layer_num):
            outC = C_structure[layer_idx]
            k = k_structure[layer_idx]
            s = s_structure[layer_idx]
            p = ((k[0]-s[0])//2, (k[1]-s[1])//2)

            if layer_idx == layer_num-1:
                self.conv_mu = nn.ConvTranspose2d(inC, outC, k, s, padding=p)
                self.conv_logvar = nn.ConvTranspose2d(inC, outC, k, s, padding=p)
                
            else:
                self.convs.append(
                    Conv2d_GLU(inC=inC, outC=outC, k=k, s=s, p=p, transpose=True)
                )
            inC = outC
            if self.vae_type in ['VAE2', 'VAE3']:
                if self.weight_sharing:
                    inC += 1
                else:
                    inC += self.style_dim
    
    def forward(self, x, one_hot=None):
        h = x
        # concate latent
        if self.vae_type in ['VAE1', 'VAE2', 'VAE3']:
            h = attach_style(h, one_hot)
        
        # ws
        for i, conv in enumerate(self.convs):
            h = conv(h)
            if self.vae_type in ['VAE2', 'VAE3']:
                if self.weight_sharing:
                    ws = self.dummy4ws(one_hot)
                    h = attach_style(h,ws)
                else:
                    h = attach_style(h,one_hot)

        h_mu = self.conv_mu(h)
        h_logvar = self.conv_logvar(h)
    
        o = reparameterize(h_mu, h_logvar)
     
        return h_mu, h_logvar, o

class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        (1, 36, 128)
        (4, 18, 32)
        (16, 9, 16)
        (4, 4, 8)
        (1, 1, 1) = (label_dim, 1, 1)
        """
        super(Discriminator, self).__init__()
        self.style_dim = kwargs.get("style_dim", 0)
        self.vae_type = kwargs.get("vae_type", '')
        assert self.vae_type in ['VAE1', 'VAE2', 'VAE3', 'MD'], "VAE type error"
        
        C_structure = [8, 16, 32, 16, 1]
        k_structure = [(4,4), (4,4), (4,4), (3,4), (1,1)]
        s_structure = [(2,2), (2,2), (2,2), (1,2), (1,1)]

        layer_num = len(C_structure)

        inC = 1
        self.convs= nn.ModuleList([])
        for layer_idx in range(layer_num):
            if self.vae_type in ['VAE1', 'VAE2', 'VAE3']:
                inC += self.style_dim
            outC = C_structure[layer_idx] 
            k = k_structure[layer_idx]
            s = s_structure[layer_idx]
            p = ((k[0]-s[0])//2, (k[1]-s[1])//2)

            if layer_idx == layer_num-1:
                self.conv_out = nn.Sequential(
                    nn.Conv2d(inC, outC, k, s, padding=p)
                )
            else:
                self.convs.append(
                    nn.Sequential(
                        Conv2d_GLU(inC=inC, outC=outC, k=k, s=s, p=p),
                        nn.Dropout(0.1)
                    )
                )
                inC = outC
        self.linear = nn.Linear(32, 1)
        
    def forward(self, x, style):
        h = x
        if self.vae_type in ['VAE1', 'VAE2', 'VAE3']:
            h = attach_style(h, style)
        for conv in self.convs:
            h = conv(h)
            if self.vae_type in ['VAE1', 'VAE2', 'VAE3']:
                h = attach_style(h, style)

        o = self.conv_out(h)
        o = o.view(-1, 32)
        o = self.linear(o)
        # o = torch.sigmoid(o)

        return o
     
class LatentClassifier(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        latent: (8, 1, 32) => (1, 8, 32)
        => (8, 4, 16) => (16, 2, 8) => (32, 1, 4) => (16, 1, 2) => (4, 1, 1)

        """
        super(LatentClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = kwargs.get("latent_dim", 0)
        self.label_num = kwargs.get("label_num", 4)
    
        self.conv1 = Conv2d_GLU(inC=1, outC=8, k=(4,4), s=(2,2), p=(1,1))
        self.conv2 = Conv2d_GLU(inC=8, outC=16, k=(4,4), s=(2,2), p=(1,1))
        self.conv3 = Conv2d_GLU(inC=16, outC=32, k=(4,4), s=(2,2), p=(1,1))
        self.conv4 = Conv2d_GLU(inC=32, outC=16, k=(3,4), s=(1,2), p=(1,1))

        self.conv_out = nn.Conv2d(16, self.label_num, (1, 2),(1, 2), padding=(0, 0))
        self.label_out = nn.LogSoftmax(dim=1)


    def forward(self, input):
        x = input.permute(0, 2, 1, 3)

        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h = self.conv4(h3)
        
        o = self.conv_out(h)
        o = self.label_out(o)
        o = o.view(o.size()[0], -1)

        return o

class DataClassifier(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        data: (1, 36, 128) -> (1, 8, 128) 
        => (8, 4, 64) => (16, 2, 32) => (32, 1, 16) => (16, 1, 8) => (4, 1, 4)
        """
        super(DataClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_num = kwargs.get("label_num", 4)

        self.conv1 = Conv2d_GLU(inC=1, outC=8, k=(4,4), s=(2,2), p=(1,1))
        self.conv2 = Conv2d_GLU(inC=8, outC=16, k=(4,4), s=(2,2), p=(1,1))
        self.conv3 = Conv2d_GLU(inC=16, outC=32, k=(4,4), s=(2,2), p=(1,1))
        self.conv4 = Conv2d_GLU(inC=32, outC=16, k=(3,4), s=(1,2), p=(1,1))

        self.conv_out = nn.Conv2d(16, self.label_num, (1, 4),(1, 2), padding=(0, 1))
        self.label_out = nn.LogSoftmax(dim=1)

    def forward(self, input):
        input = input[:,:,:8]
        h1 = self.conv1(input)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h = self.conv4(h3)

        o = self.conv_out(h)
        o = torch.sum(o, dim=(2,3))
        o = self.label_out(o)

        return o
        
class LangClassifier(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        latent: (8, 1, 32) => (1, 8, 32)
        => (8, 4, 32) => (16, 2, 32) => (32, 2, 32) => (16, 2, 32) => (144, 1, 32)

        """
        super(LangClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = kwargs.get("latent_dim", 0)
        self.label_num = kwargs.get("label_num", 4)
    
        self.conv1 = Conv2d_GLU(inC=1, outC=8, k=(4,3), s=(2,1), p=(1,1))
        self.conv2 = Conv2d_GLU(inC=8, outC=16, k=(4,3), s=(2,1), p=(1,1))
        self.conv3 = Conv2d_GLU(inC=16, outC=32, k=(3,3), s=(1,1), p=(1,1))
        self.conv4 = Conv2d_GLU(inC=32, outC=16, k=(3,3), s=(1,1), p=(1,1))

        self.conv_out = nn.Conv2d(16, self.label_num, (2, 1),(2, 1), padding=(0, 0))
        self.label_out = nn.LogSoftmax(dim=1)


    def forward(self, input):
        x = input.permute(0, 2, 1, 3)

        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h = self.conv4(h3)
        
        o = self.conv_out(h)
        o = self.label_out(o)
        o = o.permute(0, 3, 1, 2).squeeze(3)

        return o

class VAE(nn.Module):
    def __init__(self, *args, **kwargs):
        super(VAE, self).__init__()
        self.style_dim = kwargs.get("style_dim", 0)
        self.latent_dim = kwargs.get("latent_dim", 8)
        self.vae_type = kwargs.get("vae_type", '')
        self.disentanglement = kwargs.get("disentanglement",'')
        self.weight_sharing = kwargs.get("weight_sharing",True)

        self.enc = Encoder(style_dim=self.style_dim, latent_dim=self.latent_dim,vae_type=self.vae_type, weight_sharing=self.weight_sharing)
        self.dec = Decoder(style_dim=self.style_dim, latent_dim=self.latent_dim,vae_type=self.vae_type, weight_sharing=self.weight_sharing)


    def forward(self, x, one_hot_src, one_hot_tar, is_SC=False, is_CC=False):
        z_mu, z_logvar, z = self.enc(x=x, one_hot=one_hot_src)
        y_prime_mu, y_prime_logvar, y_prime = self.dec(x=z, one_hot=one_hot_tar)

        if is_SC or is_CC:
            y_z_mu, y_z_logvar, y_z = self.enc(x=y_prime, one_hot=one_hot_tar)
            
            if is_CC:
                x_prime_mu, x_prime_logvar, x_prime = self.dec(x=y_z, one_hot=one_hot_src)
                
                return y_z_mu, y_z_logvar, y_z, x_prime_mu, x_prime_logvar, x_prime

            return z, y_z
        
        return z_mu, z_logvar, z, y_prime_mu, y_prime_logvar, y_prime


