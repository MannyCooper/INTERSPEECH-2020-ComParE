import numpy as np
import matplotlib.pyplot as mp
import Extraction

train = []
devel = []
test = []
for sentence in Extraction.list_nltk_train:
    train.extend(sentence)
for sentence in Extraction.list_nltk_devel:
    devel.extend(sentence)
for sentence in Extraction.list_nltk_test:
    test.extend(sentence)
word_list_train = set(train)
word_list_devel = set(devel)
word_list_test = set(test)
word_list_train_devel = set(train+devel)
# word_list_train_devel_test = set(train+devel+test)

word_list_devel_oov = len(
    [x for x in word_list_devel if x not in word_list_train])
word_list_test_oov = len(
    [x for x in word_list_test if x not in word_list_train])

list_train_oov_sentence_devel = []
for sentence in Extraction.list_nltk_devel:
    sentence_list_devel_oov = len([x for x in sentence if x not in word_list_train])
    list_train_oov_sentence_devel.append(sentence_list_devel_oov/len(sentence))

list_train_oov_sentence_test = []
list_train_devel_oov_sentence_test = []
for sentence in Extraction.list_nltk_test:
    sentence_list_test_oov = len([x for x in sentence if x not in word_list_train])
    sentence_list_test_oov2 = len([x for x in sentence if x not in word_list_train_devel])
    list_train_oov_sentence_test.append(sentence_list_test_oov/len(sentence))
    list_train_devel_oov_sentence_test.append(sentence_list_test_oov2/len(sentence))

# print(list_train_oov_sentence_test)
list_train_devel_oov_sentence_test

tick_train = []
tick_train_1 = np.sum(list(map(lambda x: x <= 0.05, list_train_oov_sentence_test)))
tick_train_2 = np.sum(list(map(lambda x: 0.05 < x <= 0.1, list_train_oov_sentence_test)))
tick_train_3 = np.sum(list(map(lambda x: 0.1 < x <= 0.15, list_train_oov_sentence_test)))
tick_train_4 = np.sum(list(map(lambda x: 0.15 < x <= 0.2, list_train_oov_sentence_test)))
tick_train_5 = np.sum(list(map(lambda x: 0.2 < x <= 0.25, list_train_oov_sentence_test)))
tick_train_6 = np.sum(list(map(lambda x: 0.25 < x <= 0.3, list_train_oov_sentence_test)))
tick_train_7 = np.sum(list(map(lambda x: 0.3 < x , list_train_oov_sentence_test)))
tick_train = [tick_train_1,tick_train_2,tick_train_3,tick_train_4,tick_train_5,tick_train_6,tick_train_7]

tick_train_devel = []
tick_train_devel_1 = np.sum(list(map(lambda x: x <= 0.05, list_train_devel_oov_sentence_test)))
tick_train_devel_2 = np.sum(list(map(lambda x: 0.05 < x <= 0.1, list_train_devel_oov_sentence_test)))
tick_train_devel_3 = np.sum(list(map(lambda x: 0.1 < x <= 0.15, list_train_devel_oov_sentence_test)))
tick_train_devel_4 = np.sum(list(map(lambda x: 0.15 < x <= 0.2, list_train_devel_oov_sentence_test)))
tick_train_devel_5 = np.sum(list(map(lambda x: 0.2 < x <= 0.25, list_train_devel_oov_sentence_test)))
tick_train_devel_6 = np.sum(list(map(lambda x: 0.25 < x <= 0.3, list_train_devel_oov_sentence_test)))
tick_train_devel_7 = np.sum(list(map(lambda x: 0.3 < x , list_train_devel_oov_sentence_test)))
tick_train_devel = [tick_train_devel_1,tick_train_devel_2,tick_train_devel_3,tick_train_devel_4,tick_train_devel_5,tick_train_devel_6,tick_train_devel_7]


# 设置中文字体
# mp.rcParams['font.sans-serif'] = ['SimHei']
# mp.rcParams['axes.unicode_minus'] = False

mp.figure("Histogram of Story's oov"
, facecolor='white')
mp.title("Histogram of Story's oov", fontsize=16)
mp.xlabel('Percentage', fontsize=14)
mp.ylabel('Number', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':', axis='y')
x = np.arange(7)
a = mp.bar(x - 0.2, np.array(tick_train), 0.4, color='g', label="Test out of train's vocabulary", align='center')
b = mp.bar(x + 0.2, np.array(tick_train_devel), 0.4, color='b', label="Test out of train+devel's vocabulary", align='center')
# 设置标签
for i in a + b:
    h = i.get_height()
    mp.text(i.get_x() + i.get_width() / 2, h, '%d' % int(h), ha='center', va='bottom')
mp.xticks(x, ['~5%','5~10%','10~15%','15~20%','20~25%','25~30%','30%~'])

mp.legend()
mp.show()
