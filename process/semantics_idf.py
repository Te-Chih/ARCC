"""
1.提取论文的主要特征
2.将全部论文作为语料训练词向量
3.计算所有词的IDF
"""
import random
from src.featureUtils import extract_common_features
from datetime import datetime
from src.utils import *
from tqdm import tqdm
from collections import defaultdict
from gensim.models import Word2Vec, Doc2Vec


seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)


start_time = datetime.now()


def getAllPapersFeatures():
    global papers



    topic_corpus = []
    semantic_corpus = []

    # 获取论文的title、abstract, venue 的有用词
    for i, pid in enumerate(papers):
        if i % 1000 == 0:
            print(i, datetime.now() - start_time)
        paper = papers[pid]

        paper_features = extract_common_features(paper)
        semantic_corpus.append(paper_features)

    saveJson('../data/aminerEmbeding/semantic_corpus.json', semantic_corpus)


def trainWord2Vec(corpus, i, dim=100):
    data = []
    for author_feature in corpus:
        random.shuffle(author_feature)
        data.append(author_feature)

    # model = Word2Vec(data, size=dim, window=5, min_count=5, workers=20)
    model = Word2Vec(data, vector_size=dim, window=5, min_count=5, workers=20)
    model.save('data/aminerEmbeding/model_paper_{}_{}.model'.format(i, dim))


def calWordIdf():
    cropus = parseJson('../data/aminerEmbeding/semantic_corpus.json')
    idfMap = defaultdict(int)
    docNum = len(cropus)

    for doc in tqdm(cropus, desc='cal word idf'):
        wordSet = set(doc)
        for word in wordSet:
            idfMap[word] += 1

    for word in idfMap:
        idfMap[word] = np.log(docNum / idfMap[word])

    saveJson('../data/aminerEmbeding/wordIdf.json', idfMap)


if __name__ == '__main__':

    papers = parseJson('../data/aminerData/pubs_raw.json')
    #
    getAllPapersFeatures()  # 提取论文特征，构建语料库
    #
    calWordIdf()  # 计算词的逆文本频率IDF

    wordIdf = parseJson('../data/aminerEmbeding/wordIdf.json')  # 加载词IDF
    corpus = parseJson('../data/aminerEmbeding/semantic_corpus.json')

