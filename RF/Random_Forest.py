import random

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier


class RF_Classifier:

    def __init__(self, image_path, label_path, mode):
        self.image_path = image_path
        self.label_path = label_path
        self.classes = self.load_samples()[0]
        self.image = self.load_samples()[1]
        self.label = self.load_samples()[2]
        self.project = self.load_samples()[3]
        self.transform = self.load_samples()[4]
        self.mode = mode

    def load_samples(self):

        image = gdal.Open(self.image_path)
        transform = image.GetGeoTransform()
        project = image.GetProjectionRef()
        image = image.ReadAsArray()
        label = gdal.Open(self.label_path)
        label = label.ReadAsArray()

        image = np.transpose(image, [1, 2, 0])

        label = np.array(label, dtype=np.int8)
        label -= 1
        label[label < 0] = label.max() + 1

        classes = label.max()

        return classes, image, label, project, transform

    def normalize(self, image):
        image = np.transpose(image, [2, 0, 1])
        normalized_image = []
        for channels in image:
            channels = cv2.normalize(channels, None, 0, self.classes,cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            normalized_image.append(channels)

        normalized_image = np.array(normalized_image)
        image = np.transpose(normalized_image, [1, 2, 0])

        del normalized_image
        return image

    def allocate_plot_color(self):
        palette = []
        for each_class in range(self.classes):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            palette.append([r, g, b])
        palette = np.array(palette)
        return palette

    def train_rf(self):
        rows = self.image.shape[0]
        cols = self.image.shape[1]
        bands = self.image.shape[2]

        x = self.normalize(self.image)
        x = x.reshape(rows*cols, bands)
        y = self.label.ravel()

        train = np.flatnonzero(y < self.classes)
        test = np.flatnonzero(y == self.classes)

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(x[train], y[train])
        print("开始训练")
        print("训练完成, 开始预测")
        y[test] = clf.predict(x[test])
        print("运算结束")
        y = y.reshape(rows, cols)

        plt.imshow(self.allocate_plot_color()[y])
        if self.mode == "save":
            plt.savefig("Figure.png", dpi=300)
        elif self.mode =="show":
            plt.show()
        else:
            pass
        return y


    @staticmethod
    def image_writer(project, transform, image, output_path):
        driver = gdal.GetDriverByName("GTiff")
        out = driver.Create(output_path, image.shape[1], image.shape[0], gdal.GDT_Byte)
        out.SetGeoTransform(transform)
        out.SetProjection(project)
        out.GetRasterBand(1).WriteArray(image)
        out.FlushCache()
        out = None
        print("Done")

    @staticmethod
    def change_detection(image_old, label_old, image_new, label_new, label_list, result_path,mode):
#创建对象
        result_old = RF_Classifier(image_old, label_old, mode = mode)
        result_new = RF_Classifier(image_new, label_new, mode = mode)
# 类别,投影信息
        project = result_old.project
        transform = result_old.transform
        classes = result_old.classes
# 颜色分类
        palette = []
        for each_class in range(classes * (classes-1) + 1):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            palette.append([r, g, b])
        palette = np.array(palette)
        indianpines_colors = sklearn.preprocessing.minmax_scale(palette, feature_range=(0.01 , 0.99))

        result_old = result_old.train_rf()
        result_new = result_new.train_rf()
# 行列号
        rows = result_old.shape[0]
        cols = result_old.shape[1]
# 转换为numpy
        result_old = np.array(result_old, dtype=np.int8)
        result_new = np.array(result_new, dtype=np.int8)
# 变化监测
        change_detection = []
        for r in range(rows):
            sav_r = []
            for c in range(cols):
                pixel = result_new[r,c] - result_old[r,c]
                sav_r.append(pixel)
            change_detection.append(sav_r)
        change_detection = np.array(change_detection)
# 重分类
        result_old = result_old * classes + 1
        re_class =  result_old + change_detection
# 为变化的转换为0
        for each_non_change_num in range(1, classes + 1):
            non_change_num = classes * (each_non_change_num - 1) + 1
            re_class[re_class == non_change_num] = 0
# 为各变化命名
        indianpines_class_names = ["未变化"]
        for each_class in range(classes):
            for each_text in range(classes):
                if each_text != each_class:
                    indianpines_class_names.append(label_list[each_class]+ "->" + label_list[each_text])
# 图例
        patches = [mpatches.Patch(color=indianpines_colors[i], label=indianpines_class_names[i]) for i in
                   range(classes*(classes-1)+1)]

        re_class = cv2.normalize(re_class, None, 0, classes*(classes-1), cv2.NORM_MINMAX, cv2.CV_8U )
        image_show = []
        for r in range(rows):
            sav_r = []
            for c in range(cols):
                pixel = indianpines_colors[re_class[r,c]]
                sav_r.append(pixel)
            image_show.append(sav_r)
        image_show = np.array(image_show)

        plt.rcParams['font.sans-serif']=['SimHei']
        plt.legend(handles=patches, loc=5, borderaxespad=-8.7)
        plt.imshow(image_show)
        plt.savefig(result_path + "\\" + "change_detection_RF.png", dpi=300)
        RF_Classifier.image_writer(project, transform, re_class, result_path + "\\" + "change_detection_RF.tif")
if __name__ == "__main__":
    RF_Classifier.change_detection(
                                    r"../RF&SVM_Data/image1.tif",
                                    r"../RF&SVM_Data/label1.tif",
                                    r"../RF&SVM_Data/image2.tif",
                                    r"../RF&SVM_Data/label2.tif",
                                    ["植被", "水体", "建筑", "土地", "道路"],
                                    r"../RF&SVM_Data",
                                    "pass"
                                    )
