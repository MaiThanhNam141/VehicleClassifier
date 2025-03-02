import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import pickle
import os

# Danh mục hình ảnh
Categories = ['bus', 'plane', 'car', 'bicycle', 'motorcycle']

datadir = './static/images'
flat_data_arr, target_arr = [], []

# Load dữ liệu và tiền xử lý
for category in Categories:
    path = os.path.join(datadir, category)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (100, 100, 3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(category))

# Chuyển đổi dữ liệu thành DataFrame
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
df = pd.DataFrame(flat_data)
df['Target'] = target
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Chia tập dữ liệu train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0, stratify=y)

# Thiết lập tham số và huấn luyện mô hình SVM
param_grid = {'C': [1, 1000000], 'gamma': [0.0001, 1], 'kernel': ['rbf', 'poly']}
svc = svm.SVC(probability=True)
model = GridSearchCV(svc, param_grid)
model.fit(x_train, y_train)

# Lưu mô hình vào file .pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)