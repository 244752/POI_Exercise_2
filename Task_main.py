import csv
import os
from PIL import Image
from skimage import io, img_as_ubyte, feature, color
import skimage
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def conv_to_gray_sample(area, size_x, size_y):
    pictures = os.listdir(area)
    num_samples = 10  # num_samples = samples form 1 picture
    g = 0
    for i in range(0, len(pictures)):
        img = io.imread(area + '/' + pictures[i], as_gray=True)
        x = 0
        y = 0
        for p in range(0, num_samples):
            cropped_img = img[x:size_x + x, y:size_y + y]
            img_sample_conv = (cropped_img / np.max(cropped_img) * 63).astype('uint8')
            x = x + size_x
            y = y + size_y
            io.imsave("conv_" + area + "/" + area + "_grey_" + str(g + 1) + ".jpg", img_sample_conv)
            g=g+1


def create_vector(area):
    pictures = os.listdir(area)
    num_samples = 10
    vectors = []
    for i in range(0, len(pictures) * num_samples):
        img = io.imread("conv_" + area + "/" + area + "_grey_" + str(i + 1) + ".jpg", as_gray=True)
        #img_conv = (img / np.max(img) * 63).astype('uint8')
        img_conv = img_as_ubyte(img)
        feat_greyco = skimage.feature.graycomatrix(img_conv, [1], [45], levels=256, normed=True)

        contrast = round(skimage.feature.graycoprops(feat_greyco, 'contrast')[0][0], 5)
        energy = round(skimage.feature.graycoprops(feat_greyco, 'energy')[0][0], 5)
        homogeneity = round(skimage.feature.graycoprops(feat_greyco, 'homogeneity')[0][0], 5)
        correlation = round(skimage.feature.graycoprops(feat_greyco, 'correlation')[0][0], 5)
        dissimilarity = round(skimage.feature.graycoprops(feat_greyco, 'dissimilarity')[0][0], 5)
        ASM = round(skimage.feature.graycoprops(feat_greyco, 'ASM')[0][0], 5)
        vector = [area, contrast, energy, homogeneity, correlation, dissimilarity, ASM]
        vectors.append(vector)
    with open("vector.csv", "a") as file:
        np.savetxt(file, np.array(vectors), fmt="%s", delimiter=',')


def read_train_and_test():
    with open("vector.csv", 'r') as file:
        reader = csv.reader(file, delimiter=',')
        data2 = list(reader)
        data = np.array(data2)
        X = data[:, 1:]
        Y = data[:, 0]

        classifier = svm.SVC(gamma='auto')
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=6)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Wood', 'Concrete', 'Knitwear'], cmap=plt.cm.Reds)
        plt.show()
        return accuracy


conv_to_gray_sample('wood', 128, 128)
create_vector('wood')

conv_to_gray_sample('concrete', 128, 128)
create_vector('concrete')

conv_to_gray_sample('knitwear', 128, 128)
create_vector('knitwear')

acc = read_train_and_test()
print(acc)
