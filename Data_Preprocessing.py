import cv2
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import slic, mark_boundaries
import os

def im_normalize(im):
    im = im.astype(np.float32)
    min_value = im[np.unravel_index(np.argmin(im), im.shape)]
    max_value = im[np.unravel_index(np.argmax(im), im.shape)]
    im = (im - min_value)/(max_value - min_value)
    return im

def im_overlap(box1, box2):
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)

    sax = x02 - x01
    say = y02 - y01

    sbx = x12 - x11
    sby = y12 - y11

    if lx < (sax + sbx) / 2 and ly < (say + sby) / 2:
        return True
    else:
        return False

def compute_overlap(box1, box2):
    if im_overlap(box1, box2) == True:
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2

        row = min(y02, y12) - max(y01, y11)
        col = min(x02, x12) - max(x01, x11)

        S1 = (x02 - x01) * (y02 - y01)
        S2 = (x12 - x11) * (y12 - y11)
        S12 = row * col
        IoU = S12 / (S1 + S2 - S12)
        if 0 < IoU < 1:
            return 1
        else:
            return 2
    else:
        return 0


DATA_PATH = 'Data\Guangzhou'

im1 = io.imread(os.path.join(DATA_PATH, 'im1.bmp'))
im2 = io.imread(os.path.join(DATA_PATH, 'im2.bmp'))
ref = im_normalize(io.imread(os.path.join(DATA_PATH, 'ref.bmp'), as_gray=True))


Height, Width, Dim = im1.shape
ref_label = ref.flatten()

X1 = im_normalize(im1)
X2 = im_normalize(im2)

im_diff = np.sqrt(np.square(X2 - X1).sum(axis=2))
im_diff = (im_normalize(im_diff) * 255).astype(np.uint8)

DI_PATH = os.path.join(DATA_PATH, 'DI.png')
io.imsave(DI_PATH, im_diff)


seg_scale = 10
seg_image = io.imread(os.path.join(DATA_PATH, 'PCA_GZ.bmp'))

Label_Data = scio.loadmat(os.path.join(DATA_PATH, 'Seg_Label.mat'))
seg_label = Label_Data['Data']

segment_idx = np.unique(seg_label)
segment_num = len(segment_idx)
centers = np.array([np.mean(np.nonzero(seg_label==i),axis=1) for i in segment_idx])
centers = np.floor(centers).astype(np.int32)

raw_data = []
patch_size = 2 * seg_scale + 2
flag = patch_size // 2
updated_centers = (centers + flag).astype(np.int32)

for im in [im1, im2]:
    im_expand = cv2.copyMakeBorder(im, flag, flag, flag, flag, cv2.BORDER_DEFAULT)
    tmp_data = []
    for num in range(segment_num):
        x, y = updated_centers[num]
        tmp = im_expand[x - flag:x + flag, y - flag:y + flag]
        tmp_data.append(np.expand_dims(tmp, 0))
    data = np.concatenate(tmp_data, axis=0)
    raw_data.append(data)

raw_data = np.concatenate(raw_data, axis=-1)

box_coord = np.zeros((segment_num, 4))
overlap_block_degree = []

for num in range(segment_num):
    x, y = updated_centers[num]
    x0 = x - flag
    y0 = y - flag
    x1 = x + flag
    y1 = y + flag
    box_coord[num] = [x0, y0, x1, y1]

for i in range(segment_num):
    box1 = box_coord[i]
    for j in range(segment_num):
        box2 = box_coord[j]
        value = compute_overlap(box1, box2)
        overlap_block_degree.append(value)
pretext_data_label = np.array(overlap_block_degree)

non_overlop_position = np.where(pretext_data_label == 0)[0]
partial_overlap_position = np.where(pretext_data_label == 1)[0]
fully_overlop_position = np.where(pretext_data_label == 2)[0]

np.random.shuffle(non_overlop_position)
idx0 = np.round(np.linspace(0, len(non_overlop_position) - 1, 2 * len(fully_overlop_position))).astype(int)
non_overlop_location = non_overlop_position[idx0]

np.random.shuffle(partial_overlap_position)
idx1 = np.round(np.linspace(0, len(partial_overlap_position) - 1, 2 * len(fully_overlop_position))).astype(int)
partial_overlap_location = partial_overlap_position[idx1]

selected_position = np.concatenate((non_overlop_location, partial_overlap_location, fully_overlop_position), axis=0)

np.random.seed(123)
np.random.shuffle(selected_position)

stack_data = []
image_data1, image_data2 = np.split(raw_data, 2, axis=-1)

for k in range(len(selected_position)):
    position_index = selected_position[k]

    index1 = position_index // segment_num
    index2 = position_index - index1 * segment_num

    im_block1 = image_data1[index1]
    im_block2 = image_data2[index2]

    merge_data = np.concatenate((im_block1, im_block2), axis=2)
    stack_data.append(merge_data)

input_data = np.array(stack_data)
input_label = pretext_data_label[selected_position]

data_indices = np.arange(len(input_label))

ratio = 0.1
count_num = int(len(input_label) * ratio)

index_array = [i for i in range(len(input_label))]

train_indices = index_array[:count_num * 8]
val_indices = index_array[8 * count_num:]

np.save(os.path.join(DATA_PATH, 'pretext_data.npy'), input_data)
np.save(os.path.join(DATA_PATH, 'pretext_label.npy'), input_label)

np.save(os.path.join(DATA_PATH, 'data_indices.npy'), data_indices)

np.save(os.path.join(DATA_PATH, 'train_indices.npy'), train_indices)
np.save(os.path.join(DATA_PATH, 'val_indices.npy'), val_indices)

np.save(os.path.join(DATA_PATH, 'image_data.npy'), raw_data)
np.save(os.path.join(DATA_PATH, 'ref_label.npy'), ref_label)
np.save(os.path.join(DATA_PATH, 'seg_label.npy'), seg_label)

