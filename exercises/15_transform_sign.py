import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

%matplotlib qt
stop_sign = mpimg.imread('./images/stop_sign.jpg')

plt.imshow(stop_sign)
plt.scatter([112, 199, 112, 198], [95, 89, 135, 133])
plt.scatter([50, 150, 50, 150], [50, 50, 100, 100])
plt.show()

src_points = np.float32([[112, 95], [199, 89], [112, 135], [198, 133]])
dst_points = np.float32([[50, 50], [150, 50], [50, 100], [150, 100]])

M = cv2.getPerspectiveTransform(src_points, dst_points)
M_inv = cv2.getPerspectiveTransform(dst_points, src_points)

warped = cv2.warpPerspective(stop_sign, M, (stop_sign.shape[1], stop_sign.shape[0]), flags=cv2.INTER_LINEAR)

out = np.hstack((stop_sign, warped))
out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
cv2.imshow('warped', out)
cv2.waitKey()
cv2.destroyAllWindows()
