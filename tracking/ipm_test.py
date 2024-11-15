import cv2
import numpy as np

def compute_ipm_matrix(camera_matrix, dist_coeffs, rvecs, tvecs, img_size, output_size):
    # 定义地平面的 3D 点 (X, Y, Z)，假设在 z=0 的平面上
    src_points_3d = np.array([
        [-1.0,  0.0, 0],  # 左下角 (X,Y,Z)
        [ 1.0,  0.0, 0],  # 右下角 (X,Y,Z)
        [ 1.0,  2.0, 0],  # 右上角 (X,Y,Z)
        [-1.0,  2.0, 0],  # 左上角 (X,Y,Z)
    ], dtype=np.float32)
    
    # 将3D点投影到图像平面
    src_points_2d, _ = cv2.projectPoints(src_points_3d, rvecs, tvecs, camera_matrix, dist_coeffs)
    src_points_2d = np.squeeze(src_points_2d)

    # 目标图像的点（鸟瞰图），定义成一个矩形
    dst_points_2d = np.array([
        [0, output_size[1]],  # 左下角
        [output_size[0], output_size[1]],  # 右下角
        [output_size[0], 0],  # 右上角
        [0, 0]  # 左上角
    ], dtype=np.float32)
    
    # 计算透视变换矩阵
    ipm_matrix = cv2.getPerspectiveTransform(src_points_2d, dst_points_2d)
    return ipm_matrix

def apply_ipm(image, ipm_matrix, output_size):
    # 使用透视变换进行IPM
    bird_view = cv2.warpPerspective(image, ipm_matrix, output_size)
    return bird_view

# 示例：使用虚拟相机参数进行IPM变换
if __name__ == "__main__":
    camera_matrix = np.array([
        [1031.449707, 0, 957.330994],  # fx, 0, cx
        [0, 1031.363525, 534.711609],  # 0, fy, cy
        [0, 0, 1]           # 0, 0, 1
    ], dtype=np.float32)    
    # 畸变系数 (k1, k2, p1, p2, k3)
    dist_coeffs = np.array([0.005730,-0.040153,-0.000914,0.000125, 0.027661], dtype=np.float32)
    
    # 外参 (旋转向量和平移向量)
    rvecs = np.array([0, 0, 0], dtype=np.float32)  # 假设无旋转
    tvecs = np.array([0, 0, 0.3], dtype=np.float32)  # 相机距离地面 0.3米

    # 输入图像尺寸
    img_size = (1920, 1080)

    # 输出鸟瞰图尺寸
    output_size = (1920, 1080)

    # 加载图像
    image = cv2.imread('/home/jiangziben/data/people_tracking/3d/follow/Color/frame_1731417143896712696.png')

    # 计算IPM矩阵
    ipm_matrix = compute_ipm_matrix(camera_matrix, dist_coeffs, rvecs, tvecs, img_size, output_size)

    # 应用IPM变换
    bird_view = apply_ipm(image, ipm_matrix, output_size)

    # 显示结果
    cv2.imshow("Bird View", bird_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
