# -*- coding: utf-8 -*-
_author_ = "March_H"
import torch
import torch.nn as nn
import os
import json
import random
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from torchvision import transforms
import torch.nn.functional as F
from scipy.fftpack import fft,dct
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# 构建了训练集、测试集以及验证集
# 建议在run函数中只进行训练以及验证，而测试单独写文件进行
# 注意在训练过程中只需在最开头运行一次该函数，否侧会破坏原始数据
def build_datasets():
    # 假新闻共4488条，真新闻共4640条，一共9128条，取70%（6390）作为训练集，15%（1369）作为测试集和验证集
    data_path = "./datasets/weibo21/"
    fake_data_path = data_path + "fake_release_all.json"
    true_data_path = data_path + "real_release_all.json"

    news = []
    # 读取虚假新闻
    with open(fake_data_path, "r", encoding='utf-8') as f:
        fake_data = f.readlines()
    for i in fake_data:
        i = json.loads(i)
        x = {}
        x['content'], x['label'], x['category'] = i['content'], '1', i['category']
        x['img'] = data_path + "imgs/" + i['id'] + '.jpg'
        if os.path.exists(x['img']):
            news.append(x)
    # len(news): 3229

    # 读取真实新闻
    with open(true_data_path, "r", encoding='utf-8') as f:
        true_data = f.readlines()
    for i in true_data:
        i = json.loads(i)
        x = {}
        x['content'], x['label'], x['category'] = i['content'], '0', i['category']
        x['img'] = data_path + "imgs/" + i['id'] + '.jpg'
        if os.path.exists(x['img']):
            news.append(x)
    # 随机打乱
    random.shuffle(news)
    length = len(news) # 6309
    
    # 构建训练数据
    with open(data_path + "train.txt", "w", encoding="utf-8") as f:
        f.truncate()  # 清空文件
        for i in range(int(length*0.7)):
            f.write(news[i]['content'] + '\t' +
                    news[i]['img'] + '\t' +
                    news[i]['label'] + '\t' +
                    news[i]['category'] + '\n')
    # 构建验证数据
    with open(data_path + "valid.txt", "w", encoding="utf-8") as f:
        f.truncate()  # 清空文件
        for i in range(int(length*0.7), int(length*0.85)):
            f.write(news[i]['content'] + '\t' +
                    news[i]['img'] + '\t' +
                    news[i]['label'] + '\t' +
                    news[i]['category'] + '\n')
    # 构建测试数据
    with open(data_path + "test.txt", "w", encoding="utf-8") as f:
        f.truncate()  # 清空文件
        for i in range(int(length*0.85), length):
            f.write(news[i]['content'] + '\t' +
                    news[i]['img'] + '\t' +
                    news[i]['label'] + '\t' +
                    news[i]['category'] + '\n')


def word2input(texts, vocab_file, max_len):
    tokenizer = BertTokenizer(vocab_file=vocab_file)  # 基于给出的vocab文件切词
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks


# class MDRMFNDDatasets(nn.Module): 
class MDRMFNDDatasets(Dataset): 
    def __init__(self, TextPath, vocab_file, max_len, category_dict, mode='train'):
        super(MDRMFNDDatasets, self).__init__()

        # 请将传入类的初始化参数放在以下
        self.TextPath = TextPath
        self.content = []
        self.content_token_ids = []
        self.content_masks = []
        self.imgs = []
        self.dct_img = []
        self.label = []
        self.category = []
        self.vocab_file = vocab_file
        self.max_len = max_len
        self.category_dict = category_dict

        # 下面进行训练集的构建
        if mode == 'train':
            # 下面的代码进行文本数据、图像路径检索、标签以及分类的读取
            file_path = TextPath + "train.txt"
            pass
        elif mode == 'valid':
            file_path = TextPath + "valid.txt"
            pass
        elif mode == 'test':
            file_path = TextPath + "test.txt"
            pass
        with open(file_path, "r", encoding='utf-8') as f:
            self.data = f.readlines()
        self.lens = 0

        # 将构建好的数据集存进当前类中
        for x in self.data:
            x = x.strip()
            x = x.split("\t")
            img = cv2.imread(x[1])
            if np.size(img) == 1: # 如果type(img)是NoneType
                continue
            self.content.append(x[0])
            gray_img = Image.open(x[1]).convert('L') # 有时会报image file is truncated，允许了Pillow读损坏的文件，考虑exception？
            self.imgs.append(img)
            self.label.append(x[2])
            self.category.append(x[3])
            self.dct_img.append(gray_img) # 灰度图像
            self.lens += 1


        # 下面对文本数据进行分词以及映射处理
        self.content_token_ids, self.content_masks = word2input(self.content, self.vocab_file, self.max_len)

    def __len__(self):
        return self.lens
    def __getitem__(self, item):
        # 采用torch自带的Dataloader方式加载数据集，将需要取出的数据通过该函数返回即可
        # 将返回类型设置为sample
        # 暂规定Key如下：
        # TextData：存储文本信息
        # Mask：存储mask，mask为0表示当前位为pad
        # ImageData：存储相应的图像信息
        # DctData：存储图像DCT变换
        # Label：记录该数据是否为假新闻，如果为1，则为假新闻，否则为真新闻
        # category：存储该图像的类别
        # 分类类别的标号详见run.py
        sample = {}

        # 设置文本数据
        text_data = self.content_token_ids[item]
        text_data = torch.as_tensor(text_data, dtype=torch.float32)
        sample['TextData'] = text_data # len: 170

        # 设置图像数据
        # TODO:请在这里读取图像数据并将图像数据转化为tentor，存入sample中
        transform_dct = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
        ])
        sample['DctData'] = process_dct_img(transform_dct(self.dct_img[item]))
        img_data = self.imgs[item]
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB) # 从BGR格式转化为RGB格式
        img_data = transforms.Resize((224, 224))(torch.from_numpy(img_data).permute(2, 0, 1)).float()
        sample['ImageData'] = img_data # 
        
        # 设置标签数据
        label_data = self.label[item]
        label_data = torch.as_tensor(eval(label_data), dtype=torch.float32)
        sample['Label'] = label_data # 1或0

        # 设置分类数据
        category_data = self.category_dict[self.category[item]]
        category_data = torch.as_tensor(category_data, dtype=torch.float32)
        sample['Category'] = category_data # 类如tensor(4.)

        return sample
    
def process_dct_img(img):
    # size = [1, 224, 224]
    height = img.shape[1]
    width = img.shape[2]
    N = 8 
    step = int(height/N) #28

    dct_img = np.zeros((1, N*N, step*step, 1), dtype=np.float32) #[1,64,784,1]
    fft_img = np.zeros((1, N*N, step*step, 1))
    #print('dct_img:{}'.format(dct_img.shape))
    
    i = 0
    for row in np.arange(0, height, step):
        for col in np.arange(0, width, step):
            block = np.array(img[:, row:(row+step), col:(col+step)], dtype=np.float32)
            #print('block:{}'.format(block.shape))
            block1 = block.reshape(-1, step*step, 1) #[batch_size,784,1]
            dct_img[:, i,:,:] = dct(block1) #[batch_size, 64, 784, 1]

            i += 1

    #for i in range(64):
    fft_img[:,:,:,:] = fft(dct_img[:,:,:,:]).real #[batch_size,64, 784,1]
    
    fft_img = torch.from_numpy(fft_img).float() #[batch_size, 64, 784, 1]
    new_img = F.interpolate(fft_img, size=[250,1]) #[batch_size, 64, 250, 1]
    new_img = new_img.squeeze(0).squeeze(-1) #torch.size = [64, 250]
    
    return new_img   


category_dict = {
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

build_datasets()
# test = MDRMFNDDatasets("./datasets/weibo21/","./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt",
#                        170,category_dict)
# print(test.__getitem__(1))
