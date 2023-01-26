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
        with open(osp.basename(url), 'wb') as f:
            f.write(requests.get(url).content)

def test_macos_bug():
    proto = prototxt_file_name
    model = caffemodel_file_name
    npy = hull_file_name
    pts_in_hull = np.load(npy).transpose().reshape(2, 313, 1, 1).astype(np.float32)

    net = cv.dnn.readNetFromCaffe(proto, model)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts_in_hull]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, np.float32)]

    frame = np.ones((224, 224))

    H_orig, W_orig = frame.shape[:2]
    if len(frame.shape) == 2 or frame.shape[-1] == 1:
        frame = np.tile(frame.reshape(H_orig, W_orig, 1), (1, 1, 3))

    frame = frame.astype(np.float32) / 255
    frame = cv.resize(frame, (224, 224))  # resize image to network input size

    net.setInput(cv.dnn.blobFromImage(frame))
    ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))
