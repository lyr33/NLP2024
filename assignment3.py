import gensim
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import jieba
import re

class ReadFile:
    def __init__(self, root_dir, stop_words_path):
        self.root_dir = root_dir
        self.stop_words_path = stop_words_path

    def get_corpus(self):
        with open(self.stop_words_path, 'r', encoding='utf-8') as stop_words_file:
            stop_words = [line.strip() for line in stop_words_file.readlines()]

        text_dict = {}

        r1 = '[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        listdir = os.listdir(self.root_dir)

        for file_name in listdir:
            path = os.path.join(self.root_dir, file_name)
            if os.path.isfile(path) and file_name.split('.')[-1] == 'txt' and file_name != 'inf.txt':
                with open(os.path.abspath(path), "r", encoding='ansi') as file:
                    file_content = file.read()

                file_content = file_content. \
                    replace("本书来自www.cr173.com免费txt小说下载站", '')
                file_content = file_content. \
                    replace("更多更新免费电子书请关注www.cr173.com", '')
                file_content = re.sub(r1, '', file_content)
                file_content = file_content.replace("\n", '')
                file_content = file_content.replace(" ", '')
                file_content = file_content.replace('\u3000', '')

                new_words_lst = []
                split_words = list(jieba.cut(file_content))
                for word in split_words:
                    if word not in stop_words:
                        new_words_lst.append(word)

                print(file_name.split('.')[0], '总词数：', len(new_words_lst))

                text_dict[file_name.split('.')[0]] = new_words_lst

            elif os.path.isdir(path):
                print('文件路径不存在!!!!!!')
        return text_dict

read_file = ReadFile(root_dir="./jyxstxtqj_downcc.com", stop_words_path='./停词表.txt')
text_dict = read_file.get_corpus()
# 示例文本数据
# sentences = [
#     ['i', 'love', 'machine', 'learning'],
#     ['word', 'embedding', 'is', 'fun'],
#     ['natural', 'language', 'processing', 'is', 'interesting'],
#     ['deep', 'learning', 'is', 'a', 'part', 'of', 'machine', 'learning'],
# ]

# 将corpus转换为列表形式以供Word2Vec使用
sentences = list(text_dict.values())

# 训练Word2Vec模型
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 获取词向量
word_vectors = model.wv

# 打印某个词的词向量
print("Vector for '女子':", word_vectors['小姑娘'])

# 计算两个词之间的语义距离
distance = word_vectors.distance('女子', '小姑娘')
print("Distance between '女子' and '小姑娘':", distance)

# 进行聚类
words = list(word_vectors.index_to_key)
vectors = [word_vectors[word] for word in words]
kmeans = KMeans(n_clusters=2)
kmeans.fit(vectors)
labels = kmeans.labels_

# 降维以进行可视化
pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

# 可视化聚类结果
plt.scatter(result[:, 0], result[:, 1], c=labels)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()

