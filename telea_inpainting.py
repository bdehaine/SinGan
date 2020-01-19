import numpy as np
import cv2
outFile = 'Output/Inpainting/dst.png'

img = cv2.imread('Input/Harmonization/starry_night_naive.png')
mask = cv2.imread('Input/Harmonization/starry_night_naive_mask.png',0)

dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
print(mask)

#mask = np.reshape(mask,(mask.shape[0],mask.shape[1],1))
print(mask.shape)
print(dst.shape)
#mask = np.repeat(mask,3,axis=-1)

cv2.imshow('img',img)
cv2.imshow('mask',mask)
cv2.imshow('dst',dst)
res = cv2.bitwise_and(dst,dst,mask = mask)
cv2.imshow('d',res)

cv2.imwrite(outFile, dst)
cv2.waitKey(0)

cv2.destroyAllWindows()

