import numpy as np
import cv2
import sys

# read depth png
f = cv2.imread(sys.argv[1])/10.
print f.shape

# save depth in numpy format
np.save(sys.argv[1][:-3] + "npy",
        f[:,:,1].T.astype(np.float32).copy("C"))

outfn = sys.argv[1][:-3] + "dat"
f[:,:,0].T.astype(np.float32).copy("C").tofile(outfn)
