import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

# encoder / decoder 
class get_encoder_layer(nn.Module):
    def __init__(self, IN, EMBED_SIZE, Z_DIM, dropout):
        super(get_encoder_layer, self).__init__()
        self.Z_DIM = Z_DIM
        self.encoder = nn.Sequential(
            nn.Linear(IN, EMBED_SIZE),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, EMBED_SIZE)
            )
        
    def forward(self, x):
        params = self.encoder(x)
        mu, sigma = params[:, :self.Z_DIM], params[:, self.Z_DIM:]
        sigma = F.softplus(sigma) + 1e-7  # Make sigma always positive
        
        params = torch.cat((mu, sigma), 1)

        return params, mu, sigma, Independent(Normal(loc=mu, scale=sigma ), 1) # Return a factorized Normal distribution

class get_side_encoder_layer(nn.Module):
    def __init__(self, IN, EMBED_SIZE, Z_DIM, dropout):
        super(get_side_encoder_layer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(IN, EMBED_SIZE),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, Z_DIM)
            )
        
    def forward(self, x):
        return self.encoder(x)


class get_decoder_layer(nn.Module):
    def __init__(self, Z_DIM, EMBED_SIZE, OUT, dropout):
        super(get_decoder_layer, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(Z_DIM, EMBED_SIZE),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, OUT)
            )
    def forward(self, x):
        return self.decoder(x)   


######################
# MV InfoMax Trainer #
######################

class MIEstimator(nn.Module):
    def __init__(self, size1, size2, EMBED_SIZE=1024):
        super(MIEstimator, self).__init__()

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, EMBED_SIZE),
            nn.ReLU(True),
            nn.Linear(EMBED_SIZE, EMBED_SIZE),
            nn.ReLU(True),
            nn.Linear(EMBED_SIZE, 1),
        )

    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -F.softplus(-pos).mean() - F.softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1

    def get_loss(self, x1, x2):
        mi_gradient, mi_estimation =  self.forward(x1, x2)
        mi_gradient = mi_gradient.mean()
        mi_estimation = mi_estimation.mean()
        loss = - mi_gradient
        return loss




class ibLoss(nn.Module):
    def __init__(self, Z_DIM, EMBED_SIZE):
        super(ibLoss, self).__init__()
        self.Z_DIM = Z_DIM
        self.EMBED_SIZE = EMBED_SIZE
        self.mi_estimator = MIEstimator(size1=self.Z_DIM, size2=self.Z_DIM, EMBED_SIZE=self.EMBED_SIZE)
        self.EMBED_SIZE = EMBED_SIZE    # embedding 大小
        

    def forward(self, p_z1_given_v1 , p_z2_given_v2 ):         # MV InfoMax
        z1 = p_z1_given_v1.rsample()
        z2 = p_z2_given_v2.rsample()
       
        mi_gradient, mi_estimation =  self.mi_estimator(z1,z2)

        # add beta
        mi_gradient = mi_gradient.mean()

        # Computing the loss function
        loss = - mi_gradient

        return loss



class kldLoss(nn.Module):
    """
    Loss function
    """
    def __init__(self, Z_DIM):
        super(kldLoss, self).__init__()
        self.Z_DIM = Z_DIM
    def forward(self, params):
        z_mu, z_logvar = params[:, :self.Z_DIM], params[:, self.Z_DIM:]
        loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))
        return loss.sum()


class MODEL(nn.Module):
    def __init__(self, NUM_USER=100, NUM_SOURCE=100, NUM_TARGET=100,  EMBED_SIZE=100, Z_DIM=50,  \
                        SIDE_INFO_USER=2000, SIDE_INFO_ITEM=2000, dropout=0, eps_scale=0.1):
        super(MODEL, self).__init__()

        self.NUM_USER = NUM_USER      # 用户数量
        self.NUM_SOURCE = NUM_SOURCE    # source 
        self.NUM_TARGET = NUM_TARGET      # target
        self.EMBED_SIZE = EMBED_SIZE    # embedding 大小
        self.SIDE_INFO_USER = SIDE_INFO_USER   # 辅助信息大小
        self.SIDE_INFO_ITEM = SIDE_INFO_ITEM   # 辅助信息大小
        self.Z_DIM = Z_DIM
        
        self.encoder_x = get_encoder_layer(self.NUM_SOURCE, EMBED_SIZE, Z_DIM, dropout)
        self.decoder_x = get_decoder_layer(Z_DIM, EMBED_SIZE, self.NUM_SOURCE, dropout)
        # self.decoder_x = get_decoder_layer(EMBED_SIZE, EMBED_SIZE, self.NUM_SOURCE)

        self.encoder_y = get_encoder_layer(self.NUM_TARGET, EMBED_SIZE, Z_DIM, dropout)
        self.decoder_y = get_decoder_layer(Z_DIM, EMBED_SIZE, self.NUM_TARGET, dropout)
        # self.decoder_y = get_decoder_layer(EMBED_SIZE, EMBED_SIZE, self.NUM_TARGET)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU

        # side info

        self.side_info_x_user_emb = nn.Linear(self.SIDE_INFO_USER, self.EMBED_SIZE) 
        self.side_info_x_encoder = get_side_encoder_layer(self.EMBED_SIZE, self.EMBED_SIZE, Z_DIM, dropout)
        
        self.side_info_y_user_emb = nn.Linear(self.SIDE_INFO_USER, self.EMBED_SIZE) 
        self.side_info_y_encoder = get_side_encoder_layer(self.EMBED_SIZE, self.EMBED_SIZE, Z_DIM, dropout)
               
        self.eps_scale = eps_scale

    def forward(self, batch_rate_x, batch_rate_y, 
                      side_info_x_user, side_info_x_item, 
                       side_info_y_user, side_info_y_item, mode='all'):
        """
        Args:
            batch_rate_x: rating matrix -> (NUM_USER, NUM_SOURCE)
            batch_rate_y: rating matrix -> (NUM_USER, NUM_TARGET)
            side_info: index side information 
        """
        if mode=='all':  # do not split it into different forward function!
            # processing rate matrix 
            ## encoder
            feature_x_z, feature_x_mu, _, p_z_x  = self.encoder_x(self.dropout(batch_rate_x)) 
            feature_y_z, feature_y_mu, _, p_z_y = self.encoder_y(self.dropout(batch_rate_y))
            
            z_x = F.relu( feature_x_mu )
            z_y = F.relu( feature_y_mu )
            ## decoder
            preds_x = self.decoder_x(z_x)
            preds_y = self.decoder_y(z_y)   

            # processing side info
            ## encoder 
            ### side info x
            _side_info_x_user_emb = self.side_info_x_user_emb(self.dropout(side_info_x_user))
            
            side_info_feature_x_mu = self.side_info_x_encoder(_side_info_x_user_emb)
            
            ### side info y
            _side_info_y_user_emb = self.side_info_y_user_emb(self.dropout(side_info_y_user))

            side_info_feature_y_mu = self.side_info_y_encoder(_side_info_y_user_emb)

            z_contro_x = F.relu( side_info_feature_x_mu )
            z_contro_y = F.relu( side_info_feature_y_mu )         
            
            
            # Eqeual tansformation
            mapped_z_x, mapped_z_y = z_x, z_y   # equal

            preds_x2y = self.decoder_y(mapped_z_x)
            preds_y2x = self.decoder_x(mapped_z_y) 

            # generation
            generation_x = self.decoder_x(z_contro_x)   
            generation_y = self.decoder_y(z_contro_y)   

            return preds_x, preds_y, preds_x2y, preds_y2x, \
                    feature_x_z, feature_y_z, z_x, z_y, z_contro_x, z_contro_y, p_z_x, p_z_y, generation_x, generation_y

        elif mode=='sideinfo':  # do not split it into different forward function!
            # processing rate matrix 
            ## encoder
            feature_x_z, feature_x_mu, _, p_z_x  = self.encoder_x(self.dropout(batch_rate_x)) 
            feature_y_z, feature_y_mu, _, p_z_y = self.encoder_y(self.dropout(batch_rate_y))
            
            z_x = F.relu( feature_x_mu )
            z_y = F.relu( feature_y_mu )

            # processing side info
            ## encoder 
            ### side info x
            _side_info_x_user_emb = self.side_info_x_user_emb(self.dropout(side_info_x_user))
            
            side_info_feature_x_mu = self.side_info_x_encoder(_side_info_x_user_emb)
            
            ### side info y
            _side_info_y_user_emb = self.side_info_y_user_emb(self.dropout(side_info_y_user))

            side_info_feature_y_mu = self.side_info_y_encoder(_side_info_y_user_emb)

            z_contro_x = F.relu( side_info_feature_x_mu )
            z_contro_y = F.relu( side_info_feature_y_mu )         
            
            return z_x, z_y, z_contro_x, z_contro_y 

        elif mode=='get_mi_yab':        
            ## side_info_y
            _side_info_y_user_emb = self.side_info_y_user_emb(self.dropout(side_info_y_user))
            side_info_feature_y_mu = self.side_info_y_encoder(_side_info_y_user_emb)        
            z_contro_y = F.relu(side_info_feature_y_mu)   

            # decoder
            y_y = self.decoder_y(z_contro_y)

            mapped_z_y = z_contro_y
            y_x = self.decoder_x(mapped_z_y)

            return y_x, y_y, z_contro_y

        elif mode=='get_condition':        
            ## side_info_y
            _side_info_y_user_emb = self.side_info_y_user_emb(self.dropout(side_info_y_user))
            side_info_feature_y_mu = self.side_info_y_encoder(_side_info_y_user_emb)        
            z_contro_y = F.relu(side_info_feature_y_mu)   

            # decoder
            y_y = self.decoder_y(z_contro_y)

            mapped_z_y = z_contro_y
            y_x = self.decoder_x(mapped_z_y)

            return y_x, y_y, z_contro_y

        elif mode=='generation':

            _side_info_y_user_emb = self.side_info_y_user_emb(self.dropout(side_info_y_user))

            side_info_feature_y_mu = self.side_info_y_encoder(_side_info_y_user_emb)

            side_info_feature_y = self.reparameterize(side_info_feature_y_mu)

            z_contro_y = F.relu(side_info_feature_y)     
            
            y_y = self.decoder_y(z_contro_y)               
            return y_y, z_contro_y

    def reparameterize(self, mu):
        std = self.eps_scale * mu
        eps = torch.randn_like(std)
        return mu + std*eps


    

class miScore(nn.Module):
    def __init__(self,NUM_SOURCE, NUM_TARGET):
        super(miScore, self).__init__()
        self.NUM_SOURCE = NUM_SOURCE    # source
        self.NUM_TARGET = NUM_TARGET      # target
        self.aap_norm = nn.Linear(self.NUM_TARGET, self.NUM_SOURCE)   

    def forward(self, y_a, y_b):
        """
        y_a: source domain
        y_b: target domain
        """
        num_usr, H = y_a.shape
        y_b = self.aap_norm(y_b)                         # target -> source --> [B, H'] -> [B, H]
        y_b = y_b.unsqueeze(1)                                 #[B, 1(L), H]
        y_b = y_b.view([-1, H, 1]) # [B*L H 1]                 # [B, H, 1]
        ans = []
        for idx in range(num_usr):
            score = torch.matmul(y_a[idx].repeat(H,1), y_b)
            output = torch.sigmoid(score.squeeze(-1))                   # [B*L tag_num]
            ans.append(torch.sum(output,0).unsqueeze(0)/H)                                 # [1, H]
        out = torch.cat(ans, dim=0) 

        return out
