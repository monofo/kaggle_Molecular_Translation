import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from configure import *
from torch.nn.utils.rnn import pack_padded_sequence

from models.fairseq_transfomer import *
# from models.pre_transfomer import *

# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
# https://github.com/RoyalSkye/Image-Caption



def swish(x, inplace: bool = False):
    """Swish - Described originally as SiLU (https://arxiv.org/abs/1702.03118v3)
    and also as Swish (https://arxiv.org/abs/1710.05941).
    TODO Rename to SiLU with addition to PyTorch
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)


class Encoder(nn.Module):
    def __init__(self,net_type):
        super(Encoder, self).__init__()

        self.cnn = timm.create_model(net_type, pretrained=True)
        self.n_features = self.cnn.classifier.in_features


    def forward(self, x):
        bs = x.size(0)
        features = self.cnn.forward_features(x)
        return features

    def freeze_parameter(self):
        for param in self.cnn.parameters():
            param.requires_grad = False



class Net(nn.Module):

    def __init__(self,
                net_type="none",
                image_dim = 1024,
                text_dim  = 1024,
                decoder_dim = 1024,
                num_layer = 2,
                num_head = 8,
                ff_dim = 1024,
                num_pixel=10*10,
                ):

        super(Net, self).__init__()

        self.image_dim = image_dim
        self.text_dim = text_dim
        self.decoder_dim = decoder_dim
        self.num_layer = num_layer
        self.num_head = num_head
        self.ff_dim = ff_dim
        self.num_pixel = num_pixel

        self.cnn = Encoder(net_type)
        self.image_embed = nn.Sequential(
            nn.Conv2d(self.cnn.n_features,image_dim, kernel_size=1, bias=None),
            nn.BatchNorm2d(image_dim),
            Swish()
        )

        self.image_pos    = PositionEncode2D(image_dim,int(num_pixel**0.5)+1,int(num_pixel**0.5)+1) #nn.Identity() #
        self.image_encode = TransformerEncode(image_dim, ff_dim, num_head, num_layer)
        #---
        self.text_pos    = PositionEncode1D(text_dim,max_length)#nn.Identity() #
        self.token_embed = nn.Embedding(vocab_size, text_dim)
        self.text_decode = TransformerDecode(decoder_dim, ff_dim, num_head, num_layer)

        #---

        self.logit  = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

        #----
        # initialization
        self.token_embed.weight.data.uniform_(-0.05, 0.05)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-0.05, 0.05)



    @torch.jit.unused
    def forward(self, image, token, length):
        device = image.device
        batch_size = len(image)

        # encoder_padding_mask = create_padding_mask(image, num_pixel)

        #---
        image_embed = self.cnn(image)
        image_embed = self.image_embed(image_embed)
        image_embed = self.image_pos(image_embed)
        image_embed = image_embed.permute(2,3,0,1).contiguous()
        image_embed = image_embed.reshape(self.num_pixel, batch_size, self.image_dim)
        image_embed = self.image_encode(image_embed)

        # print("image: ", image_embed.shape) # torch.Size([100, 7, 1024])

        text_embed = self.token_embed(token)
        text_embed = self.text_pos(text_embed).permute(1,0,2).contiguous()

        # print("text: ", text_embed.shape) # torch.Size([300, 7, 1024])

        text_mask = np.triu(np.ones((max_length, max_length)), k=1).astype(np.uint8)
        text_mask = torch.autograd.Variable(torch.from_numpy(text_mask)==1).to(device)

        #----
        # <todo> mask based on length of token?
        # <todo> perturb mask as augmentation https://arxiv.org/pdf/2004.13342.pdf

        x = self.text_decode(text_embed, image_embed, text_mask)
        x = x.permute(1,0,2).contiguous()

        # print("decode output", x.shape) # torch.Size([7, 300, 1024]) #(batch_size, max_length, decoder_dim)

        logit = self.logit(x) # (batch_size, max_length, vocab_size)
        return logit


    @torch.jit.export
    def forward_argmax_decode(self, image):
        #--------------------------------------------------
        device = image.device
        batch_size = len(image)
        max_length = 400
        vocab_size=39
        STOI = {
            '<sos>': 36,
            '<eos>': 37,
            '<pad>': 38,
        }

        image_embed = self.cnn(image)
        image_embed = self.image_embed(image_embed)
        image_embed = self.image_pos(image_embed)
        image_embed = image_embed.permute(2,3,0,1).contiguous()
        image_embed = image_embed.reshape(self.num_pixel, batch_size, self.image_dim)
        image_embed = self.image_encode(image_embed)


        preds = torch.zeros((batch_size, max_length, vocab_size), dtype=torch.float, device=device) - 20
        token = torch.full((batch_size, max_length), STOI['<pad>'], dtype=torch.long, device=device)
        text_pos = self.text_pos.pos
        token[:, 0] = STOI['<sos>']

        # -------------------------------------
        # fast version
        #incremental_state = {}
        incremental_state = torch.jit.annotate(
            Dict[str, Dict[str, Optional[Tensor]]],
            torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}),
        )
        for t in range(max_length-1):
            #last_token = token [:,:(t+1)]
            #text_embed = self.token_embed(last_token)
            #text_embed = self.text_pos(text_embed) #text_embed + text_pos[:,:(t+1)] #

            last_token = token[:, t]
            text_embed = self.token_embed(last_token)
            text_embed = text_embed + text_pos[:, t]
            text_embed = text_embed.reshape(1, batch_size, self.text_dim)

            x = self.text_decode.forward_one(text_embed, image_embed, incremental_state)
            x = x.reshape(batch_size, self.decoder_dim)
            # print(incremental_state.keys())

            l = self.logit(x)
            preds[:, t+1] = l
            k = torch.argmax(l, -1)  # predict max
            token[:, t+1] = k
            if ((k == STOI['<eos>']) | (k == STOI['<sos>'])).all():
                break

        # if logits:
        return preds[:, 1:]
        predict = token[:, 1:]
        return predict



# loss #################################################################
def seq_cross_entropy_loss(logit, token, length):
    truth = token[:, 1:]
    L = [l - 1 for l in length]
    logit = pack_padded_sequence(logit, L, batch_first=True).data
    truth = pack_padded_sequence(truth, L, batch_first=True).data
    loss = F.cross_entropy(logit, truth, ignore_index=STOI['<pad>'])
    return loss

# check #################################################################


def run_check_net():
    net_type = "efficientnet_v2s"
    batch_size = 7
    C,H,W = 3, 320, 320
    image = torch.randn((batch_size, C, H, W))

    token  = np.full((batch_size, max_length), STOI['<pad>'], np.int64) #token
    length = np.random.randint(5,max_length-2, batch_size)
    length = np.sort(length)[::-1].copy()
    
    for b in range(batch_size):
        l = length[b]
        t = np.random.choice(vocab_size,l)
        t = np.insert(t,0,     STOI['<sos>'])
        t = np.insert(t,len(t),STOI['<eos>'])
        L = len(t)
        token[b,:L]=t

    token  = torch.from_numpy(token).long()

    #---
    net = Net(net_type)
    net.train()
    logit = net(image, token, length)

    print('vocab_size',vocab_size)
    print('max_length',max_length)
    print('')

    print(length)
    print(length.shape)
    print(token.shape)
    print(image.shape)
    print('---')

    print(logit.shape)
    print('---')

    net.eval()
    net = torch.jit.script(net)

    predict = net.forward_argmax_decode(image)
    print(predict.shape)
    print(predict)


# main #################################################################
if __name__ == '__main__':
    run_check_net()
    pass
