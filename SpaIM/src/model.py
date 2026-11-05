import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import softmax, cosine_similarity
from torch.autograd import Variable

def gram_matrix(feat):
    b,d = feat.shape
    G = torch.mm(feat, feat.t()) # b * d * d * b
    return G.div(b * d)


class mlp_simple(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.l = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, use_norm=True):
        x = self.l(x)
        if use_norm:
            x = self.norm(x)
            x = self.relu(x)
        return x


class Imputation(nn.Module):
    def __init__(self, scdim, stdim, style_dim, hidden_dims):
        super().__init__()
        h2, h1 = hidden_dims
        self.st_enc1_cont = mlp_simple(stdim, h1)
        self.st_enc2_cont = mlp_simple(h1, h2)

        self.st_enc1_style = mlp_simple(stdim, h1)
        self.st_enc2_style = mlp_simple(h1, h2)

        self.st_dec2 = mlp_simple(h2, h1)
        self.st_dec1 = mlp_simple(h1, stdim)

        self.sc_enc2_cont = mlp_simple(scdim, h2)
        self.sc_enc1_cont = mlp_simple(h2, h1)

        self.enc_style2 = mlp_simple(style_dim, h2)
        self.enc_style1 = mlp_simple(style_dim, h1)

    def forward(self, sc, st, scstyle, ststyle, istrain=1):
        if istrain:
            # generate st cont
            st_cont1 = self.st_enc1_cont(st)
            st_cont2 = self.st_enc2_cont(st_cont1)

            # generate st style
            st_style1 = self.st_enc1_style(st)
            st_style2 = self.st_enc2_style(st_style1)

            # generate sc cont
            sc_cont2 = self.sc_enc2_cont(sc)
            sc_cont1 = self.sc_enc1_cont(sc_cont2)

            # generate fake style
            fake_style2 = self.enc_style2(ststyle)
            fake_style1 = self.enc_style1(ststyle)

            # real 
            real_st_up2 = self.st_dec2(st_cont2 * st_style2)
            real_st_up1 = self.st_dec1(real_st_up2 + st_cont1 * st_style1, use_norm=False)

            # fake
            fake_st_up2 = self.st_dec2(sc_cont2 * fake_style2)
            fake_st_up1 = self.st_dec1(fake_st_up2 + sc_cont1 * fake_style1, use_norm=False)

            return {
                'st_cont1': st_cont1, 'st_cont2': st_cont2,
                'sc_cont1': sc_cont1, 'sc_cont2': sc_cont2,
                'st_style1': st_style1, 'st_style2': st_style2,
                'fake_style1': fake_style1, 'fake_style2': fake_style2,
                'st_real': real_st_up1, 'st_fake': fake_st_up1
            }

        else:
            # only have sc and ststyle
            # generate st_cont
            sc_cont2 = self.sc_enc2_cont(sc)
            sc_cont1 = self.sc_enc1_cont(sc_cont2)

            fake_style2 = self.enc_style2(ststyle)
            fake_style1 = self.enc_style1(ststyle)
 
            fake_st_up2 = self.st_dec2(sc_cont2 * fake_style2)
            fake_st_up1 = self.st_dec1(fake_st_up2 + sc_cont1 * fake_style1, use_norm=False)

            return {'st_fake': fake_st_up1}
            

class ImputeModule(nn.Module):
    def __init__(self, opt, istrain=1):
        super().__init__()
        self.opt = opt
        self.mse_loss = torch.nn.MSELoss()
        self.istrain = istrain
        self.model = Imputation(opt.sc_dim, opt.st_dim, opt.style_dim, opt.model_layers)
        if opt.parallel:
            self.model = torch.nn.DataParallel(self.model).to(torch.device('cuda'))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.loss_stat = {}

    def set_input(self, inputs, istrain=1):
        if istrain:
            self.scx = Variable(inputs['scx'].to(torch.device('cuda:%d'%(self.opt.gpu))))
            self.stx = Variable(inputs['stx'].to(torch.device('cuda:%d'%(self.opt.gpu))))
            self.sc_style = Variable(inputs['sc_style'].to(torch.device('cuda:%d'%(self.opt.gpu))))
            self.st_style = Variable(inputs['st_style'].to(torch.device('cuda:%d'%(self.opt.gpu))))
        else:
            self.scx = Variable(inputs['scx'].to(torch.device('cuda:%d'%(self.opt.gpu))))
            self.st_style = Variable(inputs['st_style'].to(torch.device('cuda:%d'%(self.opt.gpu))))

    def forward(self):
        self.model.train()
        self.out = self.model(self.scx, self.stx, None, self.st_style, istrain=True)
    
    def inference(self):
        self.model.eval()
        self.output = self.model(self.scx, None, None, self.st_style, istrain=False)
        return self.output

    @torch.no_grad()
    def diagnosis(self):
        self.model.eval()
        self.output = self.model(self.scx, self.stx, self.sc_style, self.st_style, istrain=True)
        return self.output['sc_cont'], self.output['st_cont'], self.output['sc_style'], self.output['st_style']

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def compute_loss(self):
        # compute cont_loss
        cont_loss1 = self.mse_loss(self.out['st_cont1'], self.out['sc_cont1'])
        cont_loss2 = self.mse_loss(self.out['st_cont2'], self.out['sc_cont2'])
        cont_loss = cont_loss1 + cont_loss2

        # compute style_loss
        target_g1 = gram_matrix(self.out['st_style1']).detach()
        target_g2 = gram_matrix(self.out['st_style2']).detach()
        style_loss1 = self.mse_loss(gram_matrix(self.out['fake_style1']), target_g1)
        style_loss2 = self.mse_loss(gram_matrix(self.out['fake_style2']), target_g2)
        style_loss = style_loss1 + style_loss2 

        # similarity loss between real and fake
        cs_loss1 = 1 - cosine_similarity(self.out['st_real'], self.out['st_fake'], dim=1).mean()
        cs_loss2 = 1 - cosine_similarity(self.out['st_real'], self.stx, dim=1).mean()
        cs_loss = cs_loss1 + cs_loss2

        self.loss = cont_loss + style_loss + cs_loss
        self.loss_stat = {
            'loss': self.loss.item(),
            'cont_loss': cont_loss.item(),
            'style_loss': style_loss.item(),
            'cs_loss': cs_loss.item()
        }

    def backward(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    
    def update_parameters(self):
        self.forward()
        self.compute_loss()
        self.backward()
    
    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
    
    def get_current_loss(self):
        return self.loss_stat