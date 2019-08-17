# License-Plate-Recognition
# End-to-end license plate recognition and detection model  
## Requirement  
1.tensorflow1.6.0  
2.python3.5.4  
3.opencv3.4.1  
4.numpy1.13.1   
5.xlsxwriter  
6.pillow5.0.0  
## File structure  
lib文件夹：用于存放一些python接口文件，datasets主要用于数据库的读取；layer_utils为与anchor proposal相关的接口文件；model主要为网络的config配置文件；nets为基类网络的接口，如resnet、vgg等；nms为nms的c和cuda的相关加速代码；roi_data_layer为ROI层接口文件；utils为一些辅助工具接口文件，如计时、可视化等。  
output文件夹：存放训练好的faster模型；  
img_test文件夹：存放测试图片；  
detect.py:主要用来车牌检测  
VGG16.py;主要用于车牌分类  
config.py:主要保存了识别模型的参数，比如模型的输入图像大小为120*32；  
model.py:主要存储了识别模型的结构;  
test.py：恢复低分辨率，投视角等性能库；  
main.py:调用检测分类、识别模型，完成整个车牌识别。  
## Model structure
1.Procedure flow chart  
![model](https://github.com/yangkaiqing/License-Plate-Recognition/blob/master/images/modelstructure.png)
2. Plate sort  
![sort](https://github.com/yangkaiqing/License-Plate-Recognition/blob/master/images/platesort.png)
3.CRNN  
![crnn](https://github.com/yangkaiqing/License-Plate-Recognition/blob/master/images/crnn.png)
##Result
获得了第五届研究生智慧城市挑战赛“车牌识别”题目第二名。初赛识别准确率98%.  
you can download all models need in program from https://pan.baidu.com/s/1pdCBaEKL4PCP2TwDw6Vp_w
