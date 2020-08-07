import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

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
    
x_story = []
for i in range(0,87):
    x_story.append(str(i+1))

x_perc = ['~5%','5~10%','10~15%','15~20%','20~25%','25~30%','30%~']

x=np.arange(0,87)
print(list_train_oov_sentence_test)
# train_l_devel=plt.plot(x_story,list_train_oov_sentence_devel,'ro-',label="Devel out of train's vocabulary")
train_l_test=plt.plot(x_story,list_train_oov_sentence_test,'g+-',label="Test out of train's vocabulary")
train_devel_l_test=plt.plot(x_story,list_train_devel_oov_sentence_test,'b^-',label="Test out of train+devel's vocabulary")

# plt.plot(x_story,list_train_oov_sentence_devel,'ro-',x_story,list_train_oov_sentence_test,'g+-',x_story,list_train_devel_oov_sentence_test,'b^-')

plt.plot(x_story,list_train_oov_sentence_test,'g+-',x_story,list_train_devel_oov_sentence_test,'b^-')

def to_percent(temp, position):
    return '%1.0f'%(100*temp) + '%'
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
# 数据标签
# for a,b in zip(x,list_train_oov_sentence_devel):
#     plt.text(a, b+0.01, '%1.0f' % (100*b)+'%', ha='center', va= 'bottom',fontsize=7)
plt.xticks(x,x_story,size='small')
plt.title("Story's oov")
plt.xlabel('Story ID')
plt.ylabel('Percentage')
plt.legend()
plt.show()
# print(list_train_oov_sentence_devel)
# print(word_list_devel_oov)
# print('devel_oov_percentage: {:.2%}'.format(word_list_devel_oov/len(word_list_devel)))
# print(word_list_test_oov)
# print('test_oov_percentage: {:.2%}'.format(word_list_test_oov/len(word_list_test)))
# print('\n')
# train_devel = train
# train_devel.extend(devel)
# train_devel = set(train_devel)
# word_list_test_oov_t_d = len([x for x in word_list_test if x not in train_devel])
# print(word_list_test_oov_t_d)
# print('test_oov_percentage(train+devel): {:.2%}'.format(word_list_test_oov_t_d/len(word_list_test)))

