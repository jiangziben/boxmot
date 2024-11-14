import cv2
import numpy as np
import open3d as o3d


def depth_to_pointcloud(depth_img, intrinsics, scale=1.0):
    """
    将深度图转换为点云。

    Args:
        depth_img: 输入的深度图（numpy array）。
        intrinsics: 相机内参矩阵 (3x3)。
        scale: 深度值的比例因子。

    Returns:
        point_cloud: 转换后的点云（numpy array）。
    """
    # 获取图像的尺寸
    h, w = depth_img.shape

    # 生成像素坐标网格
    i, j = np.meshgrid(np.arange(w), np.arange(h), sparse=False)

    # 根据内参矩阵解算3D坐标
    z = depth_img.astype(np.float32) / scale  # 将深度值恢复为以米为单位
    x = (i - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (j - intrinsics[1, 2]) * z / intrinsics[1, 1]

    # 在展平之前，先过滤掉无效点 (z <= 0 表示无效点)
    valid_mask = z > 0
    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]

    # 将x, y, z坐标堆叠成点云
    points = np.stack((x, y, z), axis=-1)

    return points

def save_pointcloud_as_ply(point_cloud, output_file):
    """
    保存点云数据到 PLY 文件。

    Args:
        point_cloud: 点云数据（numpy array）。
        output_file: 输出文件路径。
    """
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # 保存点云为 PLY 文件
    o3d.io.write_point_cloud(output_file, pcd)

    print(f"Point cloud saved to {output_file}")

if __name__ == "__main__":
    # 加载深度图（假设为16位深度图，单位为毫米）
    depth_image_path = '/home/jiangziben/data/people_tracking/3d/follow/Depth/frame_1731417143873207305.png'
    depth_img = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    # 相机内参矩阵 (fx, fy, cx, cy)
    intrinsics = np.array([
        [687.633179, 0, 638.220703],  # fx, 0, cx
        [0, 687.575684, 356.474426],  # 0, fy, cy
        [0, 0, 1]           # 0, 0, 1
    ])
    # 将深度图转换为点云
    point_cloud = depth_to_pointcloud(depth_img, intrinsics, scale=1000.0)  # 假设深度图的单位是毫米，需要缩放为米

    # 保存点云为PLY文件
    output_file = 'output_point_cloud.ply'
    save_pointcloud_as_ply(point_cloud, output_file)
