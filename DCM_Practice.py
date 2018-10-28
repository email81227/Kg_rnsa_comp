import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom

from os.path import join
from PublicFunctions import get_files


path = r'D:\DataSet\RSNA'

train = pd.read_csv(join(path, 'stage_1_train_labels.csv'))
train.fillna(0, inplace=True)
train.x = train.x.astype(int)
train.y = train.y.astype(int)
train.width = train.width.astype(int)
train.height = train.height.astype(int)

dcms = get_files(join(path, 'train'))

ds = pydicom.dcmread(join(path, 'train', dcms[0]))
plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
plt.show()