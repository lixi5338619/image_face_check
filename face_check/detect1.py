from scipy import misc  # pip install scipy
import tensorflow as tf  # pip install tensorflow
from face_check import detect_face
import cv2  # pip install opencv-python
import matplotlib.pyplot as plt  # pip install matplotlib

minsize = 20  # minimum size of face     最小尺寸
threshold = [0.6, 0.7, 0.7]  # three steps's threshold  阈值
factor = 0.709  # scale factor            网格参数

print('Creating and loading')
# tf.Graph().as_default() 表示将这个类实例
with tf.Graph().as_default():  # 将这个类实例，也就是新生成的图作为整个 tensorflow 运行环境的默认图
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

image_path = '1.jpg'

img = misc.imread(image_path)  # 读取图片
# 人脸检测的函数是align.detect_face.detect_face
bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
# bounding_boxes在图像中返回包围框和点。
# img:输入图像最小面尺寸:最小面尺寸  pnet, rnet, onet: caffemodel
# 阈值:阈值=[th1, th2, th3]， th1-3为三步阈值
# 因素:用于创建一个扩展的因素金字塔脸大小的检测图像中

nrof_faces = bounding_boxes.shape[0]  # 人脸数目
print('找到人脸数目为：{}'.format(nrof_faces))  # 返回检测结果

print(bounding_boxes)  # 返回关键点的坐标

# 通过上面获取的坐标来在我们的要绘制的图中进行数据标注。
crop_faces = []
for face_position in bounding_boxes:  # 遍历一下
    face_position = face_position.astype(int)  # 变量类型转换
    print(face_position[0:4])

    # 通过对角线来画矩形
    cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
    crop = img[face_position[1]:face_position[3],
           face_position[0]:face_position[2], ]

    crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC)  # cv2.resize图片缩放，参数输入是 宽×高×通道
    print(crop.shape)  # 查看矩阵结构
    crop_faces.append(crop)
    # plt.imshow(crop)
    plt.show()  # 展示

plt.imshow(img)  # 显示图片
plt.show()
