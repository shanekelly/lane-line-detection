import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import pathlib


"""
Calibrates cameras using checkerboard images to remove distortion.
"""
class CameraCalibrator:

  """
  @param calibration_input_images_dir {string}: The directory to read calibration input images from.
  @param calibration_output_images_dir {string}: The directory to write calibration output images
    to.
  @param num_corners_x {int}: The number of corners in the x direction on the checkerboard in the
    calibration input images.
  @param num_corners_y {int}: The number of corners in the y direction on the checkerboard in the
    calibration input images.
  @param camera_calibration_fname {string}: The name of the file to save / load the camera
    calibration to / from.
  """
  def __init__(self, calibration_input_images_dir, calibration_output_images_dir, num_corners_x,
      num_corners_y, camera_calibration_fname):
    self.calibration_input_images_dir = calibration_input_images_dir
    self.calibration_output_images_dir = calibration_output_images_dir
    self.num_corners_x = num_corners_x
    self.num_corners_y = num_corners_y
    self.camera_calibration_fname = camera_calibration_fname

    # Camera calibration
    self.calibration = { 'camera_matrix': [], 'distortion_coefficients': [] }
    # Detected board coordinates for all images
    self.object_points = []
    # Where the respective chessboard coordinates in object_points are in the images
    self.img_points = []


  """
  Converts numpy arrays in self.calibration to normal python lists and then serializes it to the
    specified file.
  """
  def save_calibration(self):
    cal = {}
    for k, v in self.calibration.items():
      if type(v) is np.ndarray:
        cal[k] = v.tolist()
      else:
        cal[k] = v
    json.dump(cal, open(self.camera_calibration_fname, 'w'), indent=2)
    print('Camera calibration saved to {}'.format(self.camera_calibration_fname))


  """
  De-serializes the specified file, converts any normal python lists to numpy arrays, and then
    stores it in self.calibration.
  """
  def load_calibration(self):
    cal = json.load(open(self.camera_calibration_fname, 'r'))
    for k, v in cal.items():
      if type(v) is list:
        cal[k] = np.array(v)
      else:
        cal[k] = v
    self.calibration = cal
    print('Camera calibration loaded from {}'.format(self.camera_calibration_fname))


  """
  Iterates through all the calibration input images, finds the chessboard corners, generates the
    appropriate object points and image points, then calculates the camera calibration
    (camera matrix and distortion coefficients).
  """
  def calculate_calibration(self):
    print('Finding corners...')
    # All coordinates of the corners of the chessboard (x, y, z)
    #   e.g. (0,0,0), (1,0,0), (2,0,0), ..., (0,1,0), (1,1,0), (2,1,0), ...
    board_coordinates = np.zeros((self.num_corners_x * self.num_corners_y, 3), np.float32)
    board_coordinates[:,:2] = np.mgrid[0:self.num_corners_x, 0:self.num_corners_y].T.reshape(-1,2)

    calibration_input_images = pathlib.Path(self.calibration_input_images_dir).iterdir()
    for img_path in calibration_input_images: # step through the list and search for board corners
      img_name = str(img_path)
      img = cv2.imread(img_name) # open the image
      img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale

      # Find chessboard corners
      ret, corners = cv2.findChessboardCorners(img_gray, (self.num_corners_x, self.num_corners_y), None)
      if ret == True:
        self.object_points.append(board_coordinates)
        self.img_points.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (self.num_corners_x, self.num_corners_y), corners, ret)
        #write_name = 'corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)
        cv2.imshow(img_name, img)
        cv2.waitKey(500)
    cv2.destroyAllWindows()

    # Do camera calibration given object points and image points
    _, camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(self.object_points,
        self.img_points, (img.shape[1], img.shape[0]), None, None)
    self.calibration = {
        'camera_matrix': camera_matrix,
        'distortion_coefficients': distortion_coefficients }
    print('Calculated camera calibration')


  """
  Iterates through all the calibration input images and saves the pre/post distortion images
    side-by-side in the specified output directory.
  """
  def visualize_calibration(self):
    # Make the output directory if it doesn't already exists
    calibration_output_images_path = pathlib.Path(self.calibration_output_images_dir)
    if not calibration_output_images_path.exists():
      calibration_output_images_path.mkdir(parents=True)

    calibration_input_images = pathlib.Path(self.calibration_input_images_dir).iterdir()
    for img_path in calibration_input_images:
      img_name = str(img_path)
      img = cv2.imread(img_name)
      img_undistorted = cv2.undistort(img, self.calibration['camera_matrix'],
          self.calibration['distortion_coefficients'], None, self.calibration['camera_matrix'])

      # Visualize undistortion
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
      ax1.imshow(img)
      ax1.set_title('Original Image', fontsize=30)
      ax2.imshow(img_undistorted)
      ax2.set_title('Undistorted Image', fontsize=30)
      calibration_output_image_fname = self.calibration_output_images_dir + \
          pathlib.Path(img_name).stem + '.png'
      fig.savefig(calibration_output_image_fname)
      print('Wrote calibration visualization to {}'.format(calibration_output_image_fname))


  """
  Loads the camera calibration file if it exists. Otherwise, calculates the camera calibration and
    saves it to the specified file. Then, stores images visualizing the camera calibration in the
    specified directory.
  """
  def main(self):
    if pathlib.Path(self.camera_calibration_fname).exists():
      self.load_calibration()
    else:
      self.calculate_calibration()
      self.save_calibration()
    self.visualize_calibration()


if __name__ == '__main__':
  cc = CameraCalibrator('camera_calibration/input_images/', 'camera_calibration/output_images/', 9,
      6, 'camera_calibration/camera_calibration.json')
  cc.main()
