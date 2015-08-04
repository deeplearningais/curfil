from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import cv2

def loadpcl(pclfile):
    with open(pclfile, "r") as f:
        for i in xrange(11):
            f.readline()
        x = np.loadtxt(f)
    return x

def angle_with_vertical(vec):
    a = np.sum(vec * np.array((0,1,0))) / np.linalg.norm(vec)
    print "angle: ", a
    return a

def find_up_dir(filename):
    print "Processing ", filename
    cloud = loadpcl(filename)
    points  = cloud[:, :3]
    normals = cloud[:,3:6]
    rowsum = normals.sum(1)
    idx = (-np.isnan(rowsum))
    normals2 = normals[idx, :]
    km = KMeans(n_clusters=10).fit(normals2[::5])
    labels = km.predict(normals2)
    colors = np.zeros_like(rowsum)
    colors_idx = np.zeros(idx.sum(), dtype=np.float)
    angles = map(angle_with_vertical, km.cluster_centers_)
    vertical_cluster_idx = np.argmin(angles)
    vertical_cluster = km.cluster_centers_[vertical_cluster_idx]
    vertical_cluster /= np.linalg.norm(vertical_cluster)
    print "VC: ", vertical_cluster
    dists = np.dot(points, vertical_cluster)
    for lidx, label in enumerate(np.unique(labels)):
        colors_idx[labels==label] = angles[lidx]
        if lidx == vertical_cluster_idx:
            colors_idx[labels==label] -= 4
    dists[idx] -= dists[idx].min()
    colors[idx] = colors_idx
    colors = dists
    print dists[idx].min(), dists[idx].max()
    filename = os.path.splitext(filename)[0]  # .pcl
    filename = os.path.splitext(filename)[0]  # .npy
    colors[-idx] = 0
    colors = colors.reshape(480,640).astype(np.float32)
    # colors.tofile(filename + ".height")
    assert filename.endswith("_depth")
    # save as millimeters
    out = (colors * 1000).astype(np.uint16)
    print "Output min/max: ", out.min(), out.max()
    cv2.imwrite(filename[:-6] + "_height.png", out)
    CV_LOAD_IMAGE_ANYDEPTH = 2
    # inp = cv2.imread(filename[:-6] + "_height.png", CV_LOAD_IMAGE_ANYDEPTH)
    # print "I/O dif: ", (out-inp).mean()

if __name__ == "__main__":
    find_up_dir(sys.argv[1])
    # from glob import iglob
    # from joblib import Parallel, delayed
    # images = iglob(os.path.join(sys.argv[1], '*-dpt.npy.pcl'))
    # Parallel(n_jobs=5)(delayed(find_up_dir)(f) for f in images)
