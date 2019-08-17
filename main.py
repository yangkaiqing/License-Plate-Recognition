#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import zipfile
import datetime
#import codecs
import detect as d
import test as t
import cv2
import xlsxwriter
from modelde import *
import config as cfgg
import numpy as np
from VGG16 import VGG_16
from deblurResModel import deblurPredict
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


#outpath = '../detect/detect'

#test_dir ='/cpipc/data/final/task3/Task3_车牌识别'  # 测试图片路径
test_dir ='../Task3_车牌识别'  # 测试图片路径
excel_dir = 'img_test'
workbook = xlsxwriter.Workbook('../test.xlsx')
worksheet = workbook.add_worksheet()

# if not (os.path.exists('../detect')):
#     os.makedirs('../detect/detect')
#     print('detect build!')

#  加载车牌检测模型
g1 = tf.Graph()  # 加载到Session 1的graph
d.cfg.TEST.HAS_RPN = True  # Use RPN for proposals
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.5

sess1 = tf.Session(graph=g1,config=tfconfig) # Session1

tfmodel = './output/res101/voc_2007_trainval/default/res101_faster_rcnn_iter_70000.ckpt'
if not os.path.isfile(tfmodel + '.meta'):
    raise IOError(('{:s} not found.\nDid you download the proper networks from '
                   'our server and place them properly?').format(tfmodel + '.meta'))

with sess1.as_default():
    with g1.as_default():
        net = d.resnetv1(num_layers=101)
        net.create_architecture("TEST", 6,
                                tag='default', anchor_scales=[8, 16, 32], anchor_ratios=(2.0, 2.8, 3.0))
        model_saver = tf.train.Saver()
        model_ckpt = tf.train.get_checkpoint_state(tfmodel)
        model_saver.restore(sess1, tfmodel)
        print('Loaded network {:s}'.format(tfmodel))
# 加载识别模型
loss,train_decode_result,pred_decode_result=build_network(is_training=True)
saver2 = tf.train.Saver()
sess2 = tf.Session()
ckpt2 = tf.train.latest_checkpoint(cfgg.CKPT_DIR1)
if ckpt2:
    saver2.restore(sess2,ckpt2)
    print('restore from the checkpoint{0}'.format(ckpt2))
else:
    print('failed to load ckpt')
#加载分类模型
modelsort = VGG_16('vgg16_train_val_11che_sgd_100_lossweights-51-0.01.h5')
# 去除运动模糊模型
deblur_CNN = load_model('blurry-res-weights-0048-345.953.hdf5')
# 去除散焦模糊模型
motion_CNN = load_model('m-res-weights-0004-195.218.hdf5')
worksheet.write(0, 0, '车牌号')
worksheet.write(0, 1, '颜色')
worksheet.write(0, 2, '文件名')
#worksheet.write(0, 0, '车牌号')
#worksheet.write(0, 1, '车牌颜色')
#worksheet.write(0, 2, '测试文件名')
i = 1

# def del_file(path = '../detect'): #
#     ls = os.listdir(path)
#     for i in ls:
#         c_path = os.path.join(path, i)
#         if os.path.isdir(c_path):
#             del_file(c_path)
#         else:
#             os.remove(c_path)

if (os.path.exists(test_dir)):
    files = os.listdir(test_dir)
    for file in files:

        testimg_dir = os.path.join(test_dir, file)
        excelimg_dir = os.path.join(excel_dir, file)
   #     testimg_dir = os.listdir(m)
 #       print(last_file)
 #        for file_name in last_file:
 #            finall_name = os.path.join(m, file_name)
 #            finall_names = os.listdir(finall_name)
 #            for each_file in finall_names:
 #    #            print(each_file)
 #
 #                finall_name_end = os.path.join(finall_name, each_file)
 #                finall_name_end = os.path.join(finall_name, each_file)
 #                testimg_dir = finall_name_end
        file_detect_result = []
        file_color_result = []
        path_list = os.listdir(testimg_dir)
        for k in range(len(path_list)):
            path3 = os.path.join(testimg_dir, path_list[k])  # 读入图片
            excel_path = os.path.join(excelimg_dir, path_list[k])
            detect_out, detect_out1, colorout = d.demo(sess1, net, path3, path_list[k])  # 车牌检测，返回车牌区域，以及用于车牌分类的区域和车牌颜色
            detect_out_copy = detect_out.copy()

            detect_out1 = detect_out1 / 255
            detect_out1 = cv2.resize(detect_out1, (224, 224))
            detect_out1 = np.expand_dims(detect_out1, axis=0)
            predict_img = modelsort.predict(detect_out1)

            # out = os.path.join(outpath, path_list[k])
            # cv2.imencode('.jpg', detect_out1)[1].tofile(out)
            # val_datagen = ImageDataGenerator(rescale=1. / 255)
            # validation_generator = val_datagen.flow_from_directory(
            #     '../detect',
            #     target_size=(224, 224),
            #     batch_size=1,
            #     class_mode='categorical',
            #     shuffle=False)
            # predict_img = modelsort.predict_generator(
            #     validation_generator,
            #     steps=1,
            #     verbose=0)

            acc = predict_img[0]  # 分类网络
            # # 对检测到的图片进行分类，并进行相应前处理
            _positon = np.argmax(acc)
            print('max:', _positon)
            if _positon == 0:  # 正常车牌
                detect_out = detect_out
                w = detect_out.shape[0]
                h = detect_out.shape[1]
                ration = h / w
                if ration > 2.7:  # 正常车牌细分类
                    detect_out = detect_out
                else:
                    detect_out = t.sctodc(detect_out)
            elif (_positon == 1):  # 根据分类结果进行不同的前处理
                try:
                    detect_out = t.shuzhitoushejiao(detect_out)
                except:
                    detect_out = detect_out
            elif (_positon == 2):
                try:
                    detect_out = t.toushijiao2(detect_out)
                except:
                    detect_out = detect_out
            elif (_positon == 3):
                try:
                    detect_out = t.bright1(detect_out)
                except:
                    detect_out = detect_out
            elif ((_positon == 4) or (_positon == 10)):
                try:
                    detect_out = t.bright(detect_out)
                except:
                    detect_out = detect_out

            elif (_positon == 5):
                # detect_out = detect_out
                detect_out = deblurPredict(detect_out, deblur_CNN)
            elif (_positon == 6):
                try:
                    detect_out2 = t.cuoqie(detect_out)
                    detect_out = cv2.resize(detect_out2, (120, 32))
                    print(1)
                except:
                    detect_out = cv2.resize(detect_out, (120, 32))
            elif (_positon == 7):
                try:
                    detect_out = t.xuanzhuan(detect_out)
                except:
                    detect_out = detect_out
            elif (_positon == 8):
                detect_out = deblurPredict(detect_out, motion_CNN)
                # detect_out = detect_out
            else:
                detect_out = detect_out

            c = detect_out.shape
            if len(c) == 3:
                detect_out = cv2.cvtColor(detect_out, cv2.COLOR_BGR2GRAY)
                if detect_out is None:
                    print('none')
                    detect_out = cv2.cvtColor(detect_out_copy, cv2.COLOR_BGR2GRAY)
            else:
                detect_out = detect_out
                if detect_out is None:
                    print('none1')
                    detect_out = cv2.cvtColor(detect_out_copy, cv2.COLOR_BGR2GRAY)
            detect_out = cv2.equalizeHist(detect_out)
            detect_out = cv2.resize(detect_out, (120, 32))
            img = detect_out.swapaxes(0, 1)  # 轴对称
            img = np.array(img).reshape(1, 120, 32, 1)
            ##  写入车牌颜色
            # if colorout == 'blue plate':
            #     worksheet.write(i, 1, '蓝')
            # elif colorout == 'yellow plate':
            #     worksheet.write(i, 1, '黄')
            # elif colorout == 'green plate':
            #     worksheet.write(i, 1, '绿')
            # elif colorout == 'black plate':
            #     worksheet.write(i, 1, '黑')
            # else:
            #     worksheet.write(i, 1, '白')
            val_predict = sess2.run(pred_decode_result, feed_dict={image: img})
            predit = cfg.int2label(np.argmax(val_predict, axis=2))
            print(predit[0])

            if colorout == 'blue plate':
                worksheet.write(i, 1, '蓝')
            elif colorout == 'yellow plate':
                worksheet.write(i, 1, '黄')
            elif colorout == 'green plate':
                tmppredit = str(predit[0])
                tmpone = tmppredit[-1]
                if ((tmpone == 'D') or (tmpone == 'F')):
                    worksheet.write(i, 1, '黄+绿')
                else:
                    worksheet.write(i, 1, '绿')
            elif colorout == 'black plate':
                worksheet.write(i, 1, '黑')
            else:
                worksheet.write(i, 1, '白')

   #         del_file()
            worksheet.write(i, 0, str(predit[0]))  # 写入识别结果
            #           pathtmp = codecs.decode(path3,'unicode_escape')
            #           if path3.startswith('\\u'):
            #              path3 = path3.encode('latin-1').decode('unicode_escape')

            #        strPath = str(path3, "GB2312")
            excel_path = excel_path.replace('/', '\\')
            #          path4 = 'img_tst' + '\\' + file + '\\' + file_name + '\\' + each_file + '\\' + path_list[k]
            worksheet.write(i, 2, excel_path)
            #         worksheet.write(i, 2, path3.encode('gb2312').decode('unicode_escape'))  # 写入图片路径
            i = i + 1


workbook.close()
#if os.path.exists('../cpipc05_3.xlsx'):
#    print('True')
#fzip = zipfile.ZipFile('/cpipc/cpipc05/cpipc05_3.zip', 'w', zipfile.ZIP_DEFLATED)
#fzip.write('../test.xlsx', 'test.xlsx')
#fzip.close()
#os.remove('../cpipc05_3.xlsx')
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))










