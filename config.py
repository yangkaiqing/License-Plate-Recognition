#coding:utf-8              #由于.py文件是utf-8的，所以必须有这一句
import numpy as np

learning_rate = 0.0001
momentum = 0.9
START_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2
VOCAB = {'<GO>': 0, '<EOS>': 1, '<UNK>': 2, '<PAD>': 3}  # 分别表示开始，结束，未出现的字符
VOC_IND = {}
charset = '0123456789ABCDEFGHJKLMNOPQRSTUVWXYZ京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新澳领使警学挂港'
for i in range(len(charset)):
    VOCAB[charset[i]] = i+4
for key in VOCAB:
    VOC_IND[VOCAB[key]] = key
MAX_LEN_WORD = 27  # 标签的最大长度，以PAD
VOCAB_SIZE = len(VOCAB)
RNN_UNITS = 256
EPOCH = 10000
IMAGE_WIDTH = 120
IMAGE_HEIGHT = 32
MAXIMUM__DECODE_ITERATIONS = 20
DISPLAY_STEPS = 2

CKPT_DIR1 = './models/Y/newall/'
BATCH_SIZE = 1

def label2int(label):  # label shape (num,len)标签编码
    # seq_len=[]
    target_input = np.ones((len(label), MAX_LEN_WORD), dtype=np.float32) + 2  # 初始化为全为PAD
    target_out = np.ones((len(label), MAX_LEN_WORD), dtype=np.float32) + 2  # 初始化为全为PAD
    for i in range(len(label)):
        #  seq_len.append(len(label[i]))
        target_input[i][0] = 0   # 第一个为GO，字符开始标志
        for j in range(len(label[i])):
            target_input[i][j+1] = VOCAB[label[i][j]]
            target_out[i][j] = VOCAB[label[i][j]]
        target_out[i][len(label[i])] = 1
    return target_input, target_out


def int2label(decode_label):  # 标签解码
    label = []
    for i in range(decode_label.shape[0]):
        temp = ''
        for j in range(decode_label.shape[1]):
            if VOC_IND[decode_label[i][j]] == '<EOS>':
                break
            elif decode_label[i][j] == 3:
                continue
            else:
                temp += VOC_IND[decode_label[i][j]]
        label.append(temp)
    return label

