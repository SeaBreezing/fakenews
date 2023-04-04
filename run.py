# -*- coding: utf-8 -*-
_author_ = "March_H"
import dataloader
from tqdm import tqdm
from model import MultiDomainFENDModel
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
class Run():
    def __init__(self,
                 config
                 ):
        self.configinfo = config

        self.use_cuda = config['use_cuda']
        self.model_name = config['model_name']
        self.lr = config['lr']
        self.batchsize = config['batchsize']
        self.emb_type = config['emb_type']
        self.emb_dim = config['emb_dim']
        self.max_len = config['max_len']
        self.num_workers = config['num_workers']
        self.vocab_file = config['vocab_file']
        self.early_stop = config['early_stop']
        self.bert = config['bert']
        self.root_path = config['root_path']
        self.mlp_dims = config['model']['mlp']['dims']
        self.dropout = config['model']['mlp']['dropout']
        self.seed = config['seed']
        self.weight_decay = config['weight_decay']
        self.epoch = config['epoch']
        self.save_param_dir = config['save_param_dir']

        self.train_path = self.root_path + 'train.pkl'
        self.val_path = self.root_path + 'val.pkl'
        self.test_path = self.root_path + 'test.pkl'

        self.category_dict = {
            "科技": 0,
            "军事": 1,
            "教育考试": 2,
            "灾难事故": 3,
            "政治": 4,
            "医药健康": 5,
            "财经商业": 6,
            "文体娱乐": 7,
            "社会生活": 8
        }

    def config2dict(self):
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict


    def main(self):
        #TODO:在这里添加训练所需要的代码
        #TODO:模型的定义、损失函数的定义、optimizer的定义以及训练集的加载
        # datasets要传入pytorch的Dataloader类
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        train_dataset = dataloader.MDRMFNDDatasets("./datasets/weibo21/","./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt",
                       170, dataloader.category_dict, mode = 'train')
        train_dataloader = DataLoader(
            train_dataset,
            batch_size = self.batchsize,
            sampler = RandomSampler(train_dataset),
            drop_last = True,
            pin_memory = True,
            num_workers= self.num_workers
        )
        # val_dataset = dataloader.MDRMFNDDatasets("./datasets/weibo21/","./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt",
        #                170, dataloader.category_dict, mode = 'valid')
        # test_dataset = dataloader.MDRMFNDDatasets("./datasets/weibo21/","./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt",
        #                170, dataloader.category_dict, mode = 'test')
        model = MultiDomainFENDModel(emb_dim=768,
                                     mlp_dims=384,
                                     bert='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch',
                                     dropout=0.2,
                                     emb_type='bert').to(device) # 是否需要args参数？
        loss_fn = torch.nn.BCELoss()
        # optimizer = torch.optim.Adam(params=model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for epoch in range(200):
            for step_n, batch in enumerate(tqdm(train_dataloader)):
                # img = batch['ImageData'] # 
                # text = batch['Content'] # ([170])
                # label = batch['Label'] # (0.)
                # category = batch['Category'] # (3.)
                # dct_img = batch['DctData']
                # optimizer.zero_grad()              
                pred = model(text=batch['TextData'], category=batch['Category'], label=batch['Label'], img=batch['ImageData'], dct_img=batch['DctData'])
                
            # pass
