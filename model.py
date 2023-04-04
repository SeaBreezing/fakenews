import torch
import torch.nn as nn
import numpy as np
from layers import *
from transformers import BertModel
import torch_dct as dct_2d
from torchvision import transforms
import torchvision
# dct主要写在了类外，有一个DctCNN类，以及一些类外函数；pixey domain在类内

class MultiDomainFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, bert, dropout, emb_type):
        '''
        in the MDFEND paper
        :param emb_dim: 768 in bert or 200 in w2v
        :param mlp_dims: [384]
        :param bert: default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch'
        :param dropout: 0.2
        :param emb_type: default='bert'
        '''
        super(MultiDomainFENDModel, self).__init__()
        self.domain_num = 9 # the number of domains
        self.gamma = 10
        self.num_expert = 5# the number of expert, in the MDFEND, it uses 5 experts
        self.fea_size = 256
        self.emb_type = emb_type
        if (emb_type == 'bert'):
            self.bert = BertModel.from_pretrained(bert).requires_grad_(False)

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        # build the textCNNs as experts without connection
        # each tectCNN has 768 in_channels and 64 out_channels
        expert = []
        for i in range(self.num_expert):
            expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.expert = nn.ModuleList(expert)

        # debug
        # self.gate = nn.Sequential(nn.Linear(emb_dim * 2, mlp_dims[-1]),
        #                           nn.ReLU(),
        #                           nn.Linear(mlp_dims[-1], self.num_expert),
        #                           nn.Softmax(dim=1))

        self.attention = MaskAttention(emb_dim)

        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim)
        self.specific_extractor = SelfAttentionFeatureExtract(multi_head_num=1, input_size=emb_dim,
                                                              output_size=self.fea_size)
        model_dim = 256
        kernel_sizes = [3, 3, 3]
        num_channels = [32, 64, 128]
        self.dct_img = DctCNN(model_dim,
                              0.5, # dropout = 0.5
                              kernel_sizes,
                              num_channels,
                              in_channel=128,
                              branch1_channels=[64],
                              branch2_channels=[48, 64],
                              branch3_channels=[64, 96, 96],
                              branch4_channels=[32],
                              out_channels=64)
        self.linear_dct = nn.Linear(4096, model_dim)
        self.bn_dct = nn.BatchNorm1d(model_dim)

        # pixey domain
        self.vgg = vgg(model_dim, './vgg19-dcbb9e9d.pth')
        self.linear_image = nn.Linear(4096, model_dim)
        self.bn_vgg = nn.BatchNorm1d(model_dim)
        self.drop_and_BN = 'drop-BN' # 这里会有些config
   # branch
        # GRU hidden_units:32
        # fc layer: 64, each is followed by dropout rate 0.5
        # 对pixel domain预训练时使用了数据增强

    # 所有forward（包括文本和图像）都在forward里面写
    def forward(self, **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        kwargs = {k: v.to(device) for k, v in kwargs.items()}
        img = kwargs['img']             # ([bz, 3, 224, 224])
        dct_img = kwargs['dct_img']     # ([bz, 64, 250])

        # pixey domain
        output = self.vgg(img) # ([bz, 4096])
        output = F.relu(self.linear_image(output)) # (bz, 256)
        output = self.drop_BN_layer(output, part='vgg') # (bz, 256)
        # dct_feature
        torch.cuda.empty_cache()
        dct_out = self.dct_img(dct_img) # 该类的init中定义了self.dct_img = DctCNN(...)
        dct_out = F.relu(self.linear_dct(dct_out))
        dct_out = self.drop_BN_layer(dct_out, part='dct')

        # debug
        # inputs = kwargs['content']
        # masks = kwargs['content_masks']
        # category = kwargs['category']
        # if self.emb_type == "bert":
        #     init_feature = self.bert(inputs, attention_mask=masks)[0]
        # elif self.emb_type == 'w2v':
        #     init_feature = inputs

        # feature, _ = self.attention(init_feature, masks)
        # idxs = torch.tensor([index for index in category]).view(-1, 1).cuda()
        # domain_embedding = self.domain_embedder(idxs).squeeze(1)

        # gate_input_feature = feature
        # gate_input = torch.cat([domain_embedding, gate_input_feature], dim=-1)
        # gate_value = self.gate(gate_input)

        # shared_feature = 0
        # for i in range(self.num_expert):
        #     tmp_feature = self.expert[i](init_feature)
        #     shared_feature += (tmp_feature * gate_value[:, i].unsqueeze(1))


        # return shared_feature
        return dct_out
    
    def drop_BN_layer(self, x, part='dct'):
        if part == 'dct':
            bn = self.bn_dct
        elif part == 'vgg':
            bn = self.bn_vgg
        elif part == 'bert':
            bn = self.bn_text

        if self.drop_and_BN == 'drop-BN':
            self.dropout = nn.Dropout(p = 0.5) # add myself
            x = self.dropout(x)
            x = bn(x)
        elif self.drop_and_BN == 'BN-drop':
            x = bn(x)
            x = self.dropout(x)
        elif self.drop_and_BN == 'drop-only':
            x = self.dropout(x)
        elif self.drop_and_BN == 'BN-only':
            x = bn(x)
        elif self.drop_and_BN == 'none':
            pass
        
        return x


class DctCNN(nn.Module):
    def __init__(self,
                 model_dim,
                 dropout,
                 kernel_sizes,
                 num_channels,
                 in_channel=128,
                 branch1_channels=[64],
                 branch2_channels=[48, 64],
                 branch3_channels=[64, 96, 96],
                 branch4_channels=[32],
                 out_channels=64):

        super(DctCNN, self).__init__()

        self.stem = DctStem(kernel_sizes, num_channels)

        self.InceptionBlock = DctInceptionBlock(
            in_channel,
            branch1_channels,
            branch2_channels,
            branch3_channels,
            branch4_channels,
        )

        self.maxPool = nn.MaxPool2d((1, 122))

        self.dropout = nn.Dropout(dropout)

        self.conv = ConvBNRelu2d(branch1_channels[-1] + branch2_channels[-1] +
                               branch3_channels[-1] + branch4_channels[-1],
                               out_channels,
                               kernel_size=1)

    def forward(self, dct_img):
        dct_f = self.stem(dct_img)
        x = self.InceptionBlock(dct_f)
        x = self.maxPool(x)
        x = x.permute(0, 2, 1, 3)
        x = self.conv(x)
        x = x.permute(0, 2, 1, 3)
        x = x.squeeze(-1)
        
        x = x.reshape(-1,4096)

        return x

def ConvBNRelu2d(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class DctStem(nn.Module):
    def __init__(self, kernel_sizes, num_channels):
        super(DctStem, self).__init__()
        self.convs = nn.Sequential(
            ConvBNRelu2d(in_channels=1,
                         out_channels=num_channels[0],
                         kernel_size=kernel_sizes[0]),
            ConvBNRelu2d(
                in_channels=num_channels[0],
                out_channels=num_channels[1],
                kernel_size=kernel_sizes[1],
            ),
            ConvBNRelu2d(
                in_channels=num_channels[1],
                out_channels=num_channels[2],
                kernel_size=kernel_sizes[2],
            ),
            nn.MaxPool2d((1, 2)),
        )

    def forward(self, dct_img):
        x = dct_img.unsqueeze(1)
        img = self.convs(x)
        img = img.permute(0, 2, 1, 3)

        return img

class DctInceptionBlock(nn.Module):
    def __init__(
        self,
        in_channel=128,
        branch1_channels=[64],
        branch2_channels=[48, 64],
        branch3_channels=[64, 96, 96],
        branch4_channels=[32],
    ):
        super(DctInceptionBlock, self).__init__()

        self.branch1 = ConvBNRelu2d(in_channels=in_channel,
                                    out_channels=branch1_channels[0],
                                    kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNRelu2d(in_channels=in_channel,
                         out_channels=branch2_channels[0],
                         kernel_size=1),
            ConvBNRelu2d(
                in_channels=branch2_channels[0],
                out_channels=branch2_channels[1],
                kernel_size=3,
                padding=(0, 1),
            ),
        )

        self.branch3 = nn.Sequential(
            ConvBNRelu2d(in_channels=in_channel,
                         out_channels=branch3_channels[0],
                         kernel_size=1),
            ConvBNRelu2d(
                in_channels=branch3_channels[0],
                out_channels=branch3_channels[1],
                kernel_size=3,
                padding=(0, 1),
            ),
            ConvBNRelu2d(
                in_channels=branch3_channels[1],
                out_channels=branch3_channels[2],
                kernel_size=3,
                padding=(0, 1),
            ),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1)),
            ConvBNRelu2d(in_channels=in_channel,
                         out_channels=branch4_channels[0],
                         kernel_size=1),
        )

    def forward(self, x):

        x = x.permute(0, 2, 1, 3)
        # y = x
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = out.permute(0, 2, 1, 3)

        return out
    
"""
for pixey domain
"""
class vgg(nn.Module):
    """
    obtain visual feature
    """
    def __init__(self, model_dim, pthfile):
        super(vgg, self).__init__()
        self.model_dim = model_dim
        self.pthfile = pthfile
        
        #image
        vgg_19 = torchvision.models.vgg19(pretrained=False)
        vgg_19.load_state_dict(torch.load(self.pthfile))

        self.feature = vgg_19.features
        self.classifier = nn.Sequential(*list(vgg_19.classifier.children())[:-3])
        pretrained_dict = vgg_19.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} #delect the last layer
        model_dict.update(pretrained_dict) #update 
        self.classifier.load_state_dict(model_dict) #load the new parameter
        
    def forward(self, img):
        #image
        #image = self.vgg(img) #[batch, num_ftrs]
        img = self.feature(img)
        img = img.view(img.size(0), -1)
        image = self.classifier(img)
        
        return image
