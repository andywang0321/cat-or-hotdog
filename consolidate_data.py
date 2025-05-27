from pickletools import read_decimalnl_short
import h5py
import numpy as np
import os
from PIL import Image
import zarr
import numcodecs


# Class labels
NEITHER = 0
CAT = 1
HOTDOG = 2


def read_split(split):
    cats_path: str = f"data/cats/{split}_catvsnoncat.h5"

    f = h5py.File(cats_path, "r")
    # keys: 'list_classes', 'train_set_x', 'train_set_y'

    X = f[f"{split}_set_x"][:]
    y = f[f"{split}_set_y"][:]

    hotdogs_path: str = f"/Users/andywang/Projects/cat-or-hotdog/data/hotdogs/{split}"
    hotdogs: str = hotdogs_path + "/hot_dog"
    notdogs: str = hotdogs_path + "/not_hot_dog"

    img_list = []
    label_list = []

    for cls in (hotdogs, notdogs):
        for img_name in os.listdir(cls):
            img_path: str = cls + "/" + img_name
            try:
                img = Image.open(img_path)
            except FileNotFoundError:
                print(f"Cannot find image {img_path}")
                continue
            img = img.resize((64, 64))
            img_arr = np.array(img)
            img_list.append(img_arr)
            label_list.append(HOTDOG if cls is hotdogs else NEITHER)

    dog_dset = np.array(img_list)
    dog_labels = np.array(label_list)

    images = np.concat((X, dog_dset), axis=0)
    labels = np.concat((y, dog_labels), axis=0)

    return images, labels


def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def split(a):
    a_train, a_eval, a_test = a[:1000], a[1000:1128], a[1128:]
    return a_train, a_eval, a_test


def save_zarr(images, labels, split):
    zarr_path = f"data/{split}.zarr"

    root = zarr.group(zarr_path)
    imgs = root.create_array(
        name="images",
        shape=images.shape,
        chunks=(1, images.shape[1], images.shape[2], images.shape[3]),
        dtype=images.dtype,
    )
    labs = root.create_array(name="labels", shape=labels.shape, chunks=(1,), dtype=labels.dtype)

    imgs[:] = images
    labs[:] = labels


if __name__ == "__main__":
    images_1, labels_1 = read_split("train")
    images_2, labels_2 = read_split("test")

    images = np.concat((images_1, images_2), axis=0)
    labels = np.concat((labels_1, labels_2), axis=0)

    images, labels = shuffle(images, labels)

    images_train, images_eval, images_test = split(images)
    labels_train, labels_eval, labels_test = split(labels)

    save_zarr(images_train, labels_train, "train")
    save_zarr(images_eval, labels_eval, "eval")
    save_zarr(images_test, labels_test, "test")
