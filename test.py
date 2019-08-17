import  numpy as np
import cv2
from skew_detection import *   # 用来纠正旋转
import math as m


def bright(img):   # 亮度均匀变换纠正
    m1 = 156
    m2 = 105
    m3 = 88
    a1 = 43
    a2 = 65
    a3 = 74
    imgout = img.copy()
    M1 = np.mean(img[:, :, 0])
    M2 = np.mean(img[:, :, 1])
    M3 = np.mean(img[:, :, 2])
    A1 = np.std(img[:, :, 0])
    A2 = np.std(img[:, :, 1])
    A3 = np.std(img[:, :, 2])
    imgout[:, :, 0] = a1 / A1 * (img[:, :, 0] - M1) + m1
    imgout[:, :, 1] = a2 / A2 * (img[:, :, 1] - M2) + m2
    imgout[:, :, 2] = a3 / A3 * (img[:, :, 2] - M3) + m3
    return imgout


def bright1(img):  #  亮度不均匀纠正
    imgout = img.copy()
    up = 5
    down = 5
    H = img.shape[0]
    M1 = np.mean(img[:H // 2 - up, :, 0])
    M2 = np.mean(img[:H // 2 - up, :, 1])
    M3 = np.mean(img[:H // 2 - up, :, 2])
    A1 = np.std(img[:H // 2 - up, :, 0])
    A2 = np.std(img[:H // 2 - up, :, 1])
    A3 = np.std(img[:H // 2 - up, :, 2])
    m1 = np.mean(img[H // 2 + down:, :, 0])
    m2 = np.mean(img[H // 2 + down:, :, 1])
    m3 = np.mean(img[H // 2 + down:, :, 2])
    a1 = np.std(img[H // 2 + down:, :, 0])
    a2 = np.std(img[H // 2 + down:, :, 1])
    a3 = np.std(img[H // 2 + down:, :, 2])
    imgout[:H // 2, :, 0] = a1 / A1 * (img[:H // 2, :, 0] - M1) + m1
    imgout[:H // 2, :, 1] = a2 / A2 * (img[:H // 2, :, 1] - M2) + m2
    imgout[:H // 2, :, 2] = a3 / A3 * (img[:H // 2, :, 2] - M3) + m3
    for i in range(-11, 10):
        if abs(np.mean(img[H // 2 + i + 1, :, 0]) - M1) < abs(np.mean(img[H // 2 + i + 1, :, 0]) - m1):
            imgout[H // 2 + i + 1, :, 0] = a1 / A1 * (img[H // 2 + i + 1, :, 0] - M1) + m1
            imgout[H // 2 + i + 1, :, 1] = a2 / A2 * (img[H // 2 + i + 1, :, 1] - M2) + m2
            imgout[H // 2 + i + 1, :, 2] = a3 / A3 * (img[H // 2 + i + 1, :, 2] - M3) + m3
    return imgout

def cuoqie(img):  #  错切图片纠正
    res = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    skew_h, skew_v = skew_detection(gray)
    corr_img = v_rot(img, (90 - skew_v + skew_h), img.shape, 60)
    corr_img = h_rot(corr_img, skew_h)
    hsv = cv2.cvtColor(corr_img, cv2.COLOR_BGR2HSV)
    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([124, 255, 255])
    mask = cv2.inRange(hsv, blue_lower, blue_upper)
    blurred = cv2.blur(mask, (9, 9))
    ret, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    erode = cv2.erode(closed, None, iterations=4)
    dilate = cv2.dilate(erode, None, iterations=4)
    image, contours, hierarchy = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    con = contours[0]
    rect = cv2.minAreaRect(con)
    # 矩形转换为box
    box = np.int0(cv2.boxPoints(rect))
    h1 = max([box][0][0][1], [box][0][1][1], [box][0][2][1], [box][0][3][1])
    h2 = min([box][0][0][1], [box][0][1][1], [box][0][2][1], [box][0][3][1])
    l1 = max([box][0][0][0], [box][0][1][0], [box][0][2][0], [box][0][3][0])
    l2 = min([box][0][0][0], [box][0][1][0], [box][0][2][0], [box][0][3][0])
    if h1 - h2 > 0 and l1 - l2 > 0 and h2 >= 0 and l2 >= 0:
        imgout = corr_img[h2:h1, l2:l1]
        return imgout
    else:
        return res


def xuanzhuan(image):  # 对性能库中的旋转进行测试
    res = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    (h, w) = image.shape[:2]
    if angle < -45:
        angle = -(90 + angle)

    elif abs(angle) <= 1:
        angle = angle
    else:
        angle = -(90 + angle)
    heightNew = int(w * m.fabs(m.sin(m.radians(angle))) + h * m.fabs(m.cos(m.radians(angle))))
    widthNew = int(h * m.fabs(m.sin(m.radians(angle))) + w * m.fabs(m.cos(m.radians(angle))))
    matRotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)

    matRotation[0, 2] += (widthNew - w) / 2  # 重点在这步，目前不懂为什么加这步
    matRotation[1, 2] += (heightNew - h) / 2  # 重点在这步
    center = (int(h // 2), int(w // 2))

    rotated = cv2.warpAffine(image, matRotation, (widthNew, heightNew),
                             borderValue=(255, 255, 255))
    hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([124, 255, 255])
    mask = cv2.inRange(hsv, blue_lower, blue_upper)
    blurred = cv2.blur(mask, (9, 9))
    ret, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 9))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    erode = cv2.erode(closed, None, iterations=4)
    dilate = cv2.dilate(erode, None, iterations=4)
    image, contours, hierarchy = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    con = contours[0]
    rect = cv2.minAreaRect(con)
    # 矩形转换为box
    box = np.int0(cv2.boxPoints(rect))
    h1 = max([box][0][0][1], [box][0][1][1], [box][0][2][1], [box][0][3][1])
    h2 = min([box][0][0][1], [box][0][1][1], [box][0][2][1], [box][0][3][1])
    l1 = max([box][0][0][0], [box][0][1][0], [box][0][2][0], [box][0][3][0])
    l2 = min([box][0][0][0], [box][0][1][0], [box][0][2][0], [box][0][3][0])
    if h1 - h2 > 0 and l1 - l2 > 0 and h2 >= 0 and l2 >= 0:
        imgout = rotated[h2:h1, l2:l1]
        return imgout
    else:
        return res


def difenbianlv(image):  # 低分辨率纠正
    image_out =  cv2.resize(image, (440, 140), interpolation=cv2.INTER_CUBIC)
    return image_out


def shuzhitoushejiao(img):  # 竖直投射角纠正
    res = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([124, 255, 255])
    mask = cv2.inRange(hsv, blue_lower, blue_upper)
    blurred = cv2.blur(mask, (9, 9))
    ret, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    image, contours, hierarchy = cv2.findContours(closed.copy(), 1, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    # 矩形转换为box
    box = np.int0(cv2.boxPoints(rect))
    h1 = max([box][0][0][1], [box][0][1][1], [box][0][2][1], [box][0][3][1])
    h2 = min([box][0][0][1], [box][0][1][1], [box][0][2][1], [box][0][3][1])
    l1 = max([box][0][0][0], [box][0][1][0], [box][0][2][0], [box][0][3][0])
    l2 = min([box][0][0][0], [box][0][1][0], [box][0][2][0], [box][0][3][0])
    if h1 - h2 > 0 and l1 - l2 > 0 and h2 >= 0 and l2 >= 0:
        temp = img[h2:h1, l2:l1]
    else:
        temp = img.copy()
    p = []
    l = len(contours[0])
    for i in range(l):
        if abs(contours[0][i][0][1] - h1) <= 1:
            pp = contours[0][i][0]
            p.append(pp)
    p = np.array(p)

    minpoint = min(p[:, 0])
    maxpoint = max(p[:, 0])
    maxboxvalu = box.max()
    # print(maxboxvalu )
    for x in range(4):
        if box[x][1] != 0:
            secondmaxboxvalue = box[x][1]
            break
    pts_src = np.array([[maxboxvalu, secondmaxboxvalue], [0, secondmaxboxvalue], [0, 0], [maxboxvalu, 0]])
    pts_dst = np.array([[maxboxvalu, secondmaxboxvalue], [0, secondmaxboxvalue], [minpoint, h2], [maxpoint, h2]])
    # pts_dst = np.array([[0,W], [H,0], [H,minpoint], [H,maxpoint]])
    h, status = cv2.findHomography(pts_src, pts_dst)
    im_out = cv2.warpPerspective(temp, h, (h2, l2), borderValue=(255, 255, 255))

    hsv = cv2.cvtColor(im_out, cv2.COLOR_BGR2HSV)
    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([124, 255, 255])
    mask = cv2.inRange(hsv, blue_lower, blue_upper)
    blurred = cv2.blur(mask, (9, 9))
    ret, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    image, contours, hierarchy = cv2.findContours(closed.copy(), 1, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    # 矩形转换为box
    box = np.int0(cv2.boxPoints(rect))
    h1 = max([box][0][0][1], [box][0][1][1], [box][0][2][1], [box][0][3][1])
    h2 = min([box][0][0][1], [box][0][1][1], [box][0][2][1], [box][0][3][1])
    l1 = max([box][0][0][0], [box][0][1][0], [box][0][2][0], [box][0][3][0])
    l2 = min([box][0][0][0], [box][0][1][0], [box][0][2][0], [box][0][3][0])
    if h1 - h2 > 0 and l1 - l2 > 0 and h2 >= 0 and l2 >= 0:
        temp = im_out[h2:h1, l2:l1]
        return temp
    else:
        return res


def toushijiao2(img):   # 水平透视角
    res = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([124, 255, 255])
    mask = cv2.inRange(hsv, blue_lower, blue_upper)
    blurred = cv2.blur(mask, (9, 9))
    ret, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    image, contours, hierarchy = cv2.findContours(closed.copy(), 1, cv2.CHAIN_APPROX_SIMPLE)

    rect = cv2.minAreaRect(contours[0])
    # box = np.int0(cv2.boxPoints(rect))
    box = np.int0(cv2.boxPoints(rect))

    h1 = max([box][0][0][1], [box][0][1][1], [box][0][2][1], [box][0][3][1])
    h2 = min([box][0][0][1], [box][0][1][1], [box][0][2][1], [box][0][3][1])
    l1 = max([box][0][0][0], [box][0][1][0], [box][0][2][0], [box][0][3][0])
    l2 = min([box][0][0][0], [box][0][1][0], [box][0][2][0], [box][0][3][0])
    if h1 - h2 > 0 and l1 - l2 > 0 and h2 >= 0 and l2 >= 0:
        temp = img[h2:h1, l2:l1]
    else:
        temp = img.copy()

    l = len(contours[0])
    p = []
    for i in range(l):
        if abs(contours[0][i][0][0] - l2) <= 1:
            pp = contours[0][i][0]
            p.append(pp)
    p = np.array(p)
    minpoint = min(p[:, 1])
    maxpoint = max(p[:, 1])

    for x in range(4):
        if box[x][1] != 0:
            secondmaxboxvalue = box[x][1]
            break
    pts_src = np.array([box[0], box[1], box[2], box[3]])
    pts_dst = np.array(
        [box[0], box[1], [l1, minpoint + 2], [l1, maxpoint - 2]])
    h, status = cv2.findHomography(pts_src, pts_dst)
    im_out = cv2.warpPerspective(temp, h, (h2, l2), borderValue=(255, 255, 255))
    hsv = cv2.cvtColor(im_out, cv2.COLOR_BGR2HSV)
    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([124, 255, 255])
    mask = cv2.inRange(hsv, blue_lower, blue_upper)
    blurred = cv2.blur(mask, (9, 9))
    ret, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    image, contours, hierarchy = cv2.findContours(closed.copy(), 1, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    # 矩形转换为box
    box = np.int0(cv2.boxPoints(rect))
    h1 = max([box][0][0][1], [box][0][1][1], [box][0][2][1], [box][0][3][1])
    h2 = min([box][0][0][1], [box][0][1][1], [box][0][2][1], [box][0][3][1])
    l1 = max([box][0][0][0], [box][0][1][0], [box][0][2][0], [box][0][3][0])
    l2 = min([box][0][0][0], [box][0][1][0], [box][0][2][0], [box][0][3][0])
    if h1 - h2 > 0 and l1 - l2 > 0 and h2 >= 0 and l2 >= 0:
        temp = im_out[h2:h1, l2:l1]
        return temp
    else:
        return res


def sctodc(img):  # 对双层车牌进行分割，并拼接
    h = img.shape[0]
    w = img.shape[1]
    Var = []
    Grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for x in range(h // 3, h // 2):
        v = np.var(Grayimg[x, 10:w - 10])
        Var.append(v)
    minposition = Var.index(min(Var))
    img1 = img[:minposition + h // 3, :, :]
    img2 = img[minposition + h // 3:, :, :]
    img1 = cv2.resize(img1, (w, h // 2), interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, (w, h // 2), interpolation=cv2.INTER_CUBIC)
    img3 = img1[:, 20:-25]
    img4 = img2[:, :]
    Grayimg4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    Mean = []
    for i in range(3, 12):
        v = np.mean(Grayimg4[:, i])
        Mean.append(v)
    maxposition = Mean.index(max(Mean))
    img44 = img4[:, maxposition + 3:-5]
    imgout = np.zeros((img3.shape[0], img3.shape[1] + img44.shape[1], 3), dtype=np.uint8)
    imgout[:, :img3.shape[1]] = img3
    imgout[:, img3.shape[1]:] = img44
    return imgout