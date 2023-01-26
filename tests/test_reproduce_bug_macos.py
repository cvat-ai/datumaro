import os.path as osp
import requests
import numpy as np
import cv2 as cv

prototxt_file_name = "colorization_deploy_v2.prototxt"
caffemodel_file_name = "colorization_release_v2.caffemodel"
hull_file_name = "pts_in_hull.npy"

download_files = [
    f"https://raw.githubusercontent.com/richzhang/colorization/a1642d6ac6fc80fe08885edba34c166da09465f6/colorization/models/{prototxt_file_name}",
    f"http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/{caffemodel_file_name}",
    f"https://raw.githubusercontent.com/richzhang/colorization/a1642d6ac6fc80fe08885edba34c166da09465f6/colorization/resources/{hull_file_name}",
]

def download_model(files):
    for url in files:
        filename = osp.basename(url)
        if osp.isfile(filename):
            continue

        with open(filename, 'wb') as f:
            f.write(requests.get(url).content)

def test_macos_bug():
    download_model(download_files)

    proto = prototxt_file_name
    model = caffemodel_file_name
    npy = hull_file_name
    pts_in_hull = np.load(npy).transpose().reshape(2, 313, 1, 1).astype(np.float32)

    net = cv.dnn.readNetFromCaffe(proto, model)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts_in_hull]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, np.float32)]

    frame = np.ones((224, 224), dtype=np.float32)
    # frame = np.ones((24, 36), dtype=np.uint8)

    # H_orig, W_orig = frame.shape[:2]
    # if len(frame.shape) == 2 or frame.shape[-1] == 1:
    #     frame = np.tile(frame.reshape(H_orig, W_orig, 1), (1, 1, 3))


    # frame = frame.astype(np.float32) / 255
    # img_l = rgb2lab(frame)  # get L from Lab image
    # img_rs = cv.resize(img_l, (224, 224))  # resize image to network input size
    # img_l_rs = img_rs - 50  # subtract 50 for mean-centering

    net.setInput(cv.dnn.blobFromImage(frame))
    net.forward()


def rgb2lab(frame: np.ndarray) -> np.ndarray:
    # Use of the OpenCV version sometimes leads to hangs
    y_coeffs = np.array([0.212671, 0.715160, 0.072169], dtype=np.float32)
    frame = np.where(frame > 0.04045, np.power((frame + 0.055) / 1.055, 2.4), frame / 12.92)
    y = frame @ y_coeffs.T
    L = np.where(y > 0.008856, 116 * np.cbrt(y) - 16, 903.3 * y)
    return L
