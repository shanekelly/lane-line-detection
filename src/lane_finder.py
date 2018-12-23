import cv2
import matplotlib.pyplot as plt
import numpy as np

from camera_calibrator import CameraCalibrator


class LaneFinder:

  @staticmethod
  def get_road_shaped_points(img_size, top_width_pct=0.100, bot_width_pct=1.000, height_pct=0.375):
    mid_x = img_size[0] // 2
    top_width = int(img_size[0] * top_width_pct)
    bot_width = int(img_size[0] * bot_width_pct)
    height = int(img_size[1] * height_pct)

    bot_left = (mid_x - bot_width // 2, img_size[1])
    top_left = (mid_x - top_width // 2, img_size[1] - height)
    bot_right = (mid_x + bot_width // 2, img_size[1])
    top_right = (mid_x + top_width // 2, img_size[1] - height)
    points = np.float32([bot_left, top_left, bot_right, top_right])
    return points


  @staticmethod
  def get_rectangle_shaped_points(img_size, width_pct=1.000, height_pct=1.000):
    mid_x = img_size[0] // 2
    width = int(img_size[0] * width_pct)
    height = int(img_size[1] * height_pct)

    bot_left = (mid_x - width // 2, img_size[1])
    top_left = (mid_x - width // 2, img_size[1] - height)
    bot_right = (mid_x + width // 2, img_size[1])
    top_right = (mid_x + width // 2, img_size[1] - height)
    points = np.float32([bot_left, top_left, bot_right, top_right])
    return points


  @staticmethod
  def draw_points_on_img(img, points, radius, color):
    for point in points:
      cv2.circle(img, (point[0], point[1]), radius, color, -1)


  def __init__(self, img_fname):
    self.img_fname = img_fname

    # Camera calibration
    self.calibration = { 'camera_matrix': [], 'distortion_coefficients': [] }
    self.get_calibration()


  def get_calibration(self):
    cc = CameraCalibrator('camera_calibration/input_images/', 'camera_calibration/output_images/',
        9, 6, 'camera_calibration/camera_calibration.json')
    cc.main()
    self.calibration = cc.calibration


  def undistort(self, img_distorted):
    img_undistorted = cv2.undistort(img_distorted, self.calibration['camera_matrix'],
        self.calibration['distortion_coefficients'], None,  self.calibration['camera_matrix'])
    return img_undistorted


  def warp(self, img_unwarped):
    img_size = (img_unwarped.shape[1], img_unwarped.shape[0])
    source_points = self.get_road_shaped_points(img_size)
    destination_points = self.get_rectangle_shaped_points(img_size)
    M = cv2.getPerspectiveTransform(source_points, destination_points)
    img_warped = cv2.warpPerspective(img_unwarped, M, img_size)
    self.draw_points_on_img(img_unwarped, destination_points, 20, (0, 255, 0))
    self.draw_points_on_img(img_unwarped, source_points, 10, (255, 0, 0))
    return img_warped


  def main(self):
    img = cv2.cvtColor(cv2.imread(self.img_fname), cv2.COLOR_BGR2RGB)
    img_undistorted = self.undistort(img)
    img_warped = self.warp(img_undistorted)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()
    fontsize = 30
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=fontsize)
    ax1.set_axis_off()
    ax2.imshow(img_undistorted)
    ax2.set_title('Undistorted Image', fontsize=fontsize)
    ax2.set_axis_off()
    ax3.imshow(img_warped)
    ax3.set_title('Warped Image', fontsize=fontsize)
    ax3.set_axis_off()
    plt.show()


if __name__ == '__main__':
  lf = LaneFinder('media/test_input_images/straight_lines_1.jpg')
  lf.main()
