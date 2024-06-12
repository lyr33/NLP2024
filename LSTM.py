import jieba
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import random
import re
import time
import numpy as np

def read_text(path):
    # root_dir = './jyxstxtqj_downcc.com/'
    # path = os.path.join(root_dir, '射雕英雄传.txt')
    with open(os.path.abspath(path), "r", encoding='ansi') as file:
        file_content = file.read()

    file_content = file_content. \
        replace("本书来自www.cr173.com免费txt小说下载站", '')
    file_content = file_content. \
        replace("更多更新免费电子书请关注www.cr173.com", '')
    file_content = file_content. \
        replace("----〖新语丝电子文库(www.xys.org)〗", '')

    file_content = file_content.replace("\n", '')
    file_content = file_content.replace("\t", '')
    file_content = file_content.replace(" ", '')
    file_content = file_content.replace('\u3000', '')
    file_content = file_content.replace('」', '”')
    file_content = file_content.replace('「', '“')
    file_content = file_content.replace('□', '')
    file_content = file_content.replace('』', '’')
    file_content = file_content.replace('『', '‘')
    # print(type(file_content))
    # print(len(file_content))
    text = list(jieba.cut(file_content))
    print('经过预处理后文章总词数：', len(text))
    return text

word_lst = read_text(path='./jyxstxtqj_downcc.com/射雕英雄传.txt')

# 构建字典映射每个中文字到索引的关系
word2index = {}

for word in word_lst:
    if word not in word2index:
        word2index[word] = len(word2index)

index2word = {index: word for word, index in word2index.items()}

# 将中文转换为索引
index_lst = [word2index[word] for word in word_lst]


class LSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        outputs = self.fc_out(outputs)

        return outputs

vocab_size = len(word2index) # 词典中总共的词数
embed_size = 30 # 每个词语嵌入特征数
hidden_size = 512 # LSTM的每个时间步的每一层的神经元数量
num_layers = 2 # LSTM的每个时间步的隐藏层层数

max_epoch = 30
batch_size = 16
learning_rate = 0.001
sentence_len = 20
train_lst = [i for i in range(0,10000)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = LSTM(vocab_size, embed_size, hidden_size, num_layers, vocab_size).to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def tensor_to_str(index2word_dict, class_tensor):
    # 将张量转换为字符串
    class_lst = list(class_tensor)
    words = [index2word_dict[int(index)] for index in class_lst]

    # 将列表中的词语连接为一个字符串
    sentence = ''.join(words)
    return sentence


test_model = torch.load('./trained_model/lstm2.pth').to('cpu')

generate_length = 100
test_set = [index_lst[i:i + sentence_len] for i in range(20000, 30000, 1000)]
target_set = [index_lst[i:i + sentence_len + generate_length] for i in range(20000, 30000, 1000)]

for i in range(0, len(test_set)):
    generate_lst = []
    generate_lst.extend(test_set[i])
    for j in range(0, generate_length):
        inputs = torch.tensor(generate_lst[-sentence_len:])  # 选取生成词语列表的最后sentence_len个元素作为下一次模型的输入
        outputs = test_model(inputs)

        predicted_class = torch.argmax(outputs, dim=-1)

        generate_lst.append(int(predicted_class[-1]))

    input_sentence = tensor_to_str(index2word, test_set[i])
    generate_sentence = tensor_to_str(index2word, generate_lst)
    target_sentence = tensor_to_str(index2word, target_set[i])
    # 打印生成结果
    print('测试结果', i)
    print()
    print('初始输入句：\n', input_sentence)
    print()
    print('模型生成句：\n', generate_sentence)
    print()
    print('期待生成句：\n', target_sentence)
    print()
    print('=' * 50)