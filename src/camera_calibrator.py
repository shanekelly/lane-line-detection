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
  @param {list} l - A list that contains elements of any type (including more nested lists).
  @returns {type} - The type of the element inside of the (possibly many nested) list.
  """
  @staticmethod
  def get_base_type_of_list(l):
    element = l
    while type(element) is list:
      element = element[0]
    return type(element)


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
  Converts numpy arrays in self.calibration to normal python lists and then serializes it to
    `self.camera_calibration_fname`.
  """
  def save_calibration(self):
    cal = {}
    for key, val in self.calibration.items():
      if type(val) is np.ndarray:
        cal[key] = val.tolist()
      else:
        cal[key] = val
    json.dump(cal, open(self.camera_calibration_fname, 'w'), indent=2)
    print('Camera calibration saved to {}'.format(self.camera_calibration_fname))


  """
  De-serializes the file at `self.camera_calibration_fname`, converts any normal python lists to
    numpy arrays, and then stores it in self.calibration.
  """
  def load_calibration(self):
    cal = json.load(open(self.camera_calibration_fname, 'r'))
    for key, val in cal.items():
      if type(val) is list: # load all lists as numpy arrays
        base_type = self.get_base_type_of_list(val) # type of the elements inside of any nested lists
        if base_type is float: # some camera calibration maps need to be float32 (not float64)
          cal[key] = np.array(val).astype(np.float32)
        else:
          cal[key] = np.array(val)
      else:
        cal[key] = val
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
    img_shape = img.shape[:2][::-1] # store as (x size, y size)
    _, camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(self.object_points,
        self.img_points, img_shape, None, None)
    map_x, map_y = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, None,
        camera_matrix, img_shape, 5)
    self.calibration = {
        'camera_matrix': camera_matrix,
        'distortion_coefficients': distortion_coefficients,
        'image_shape': img_shape,
        'map_x': map_x.astype(np.float32), # convert from float64 to float32 to avoid weird errors
        'map_y': map_y.astype(np.float32) }
    print('Calculated camera calibration.')


  """
  Iterates through all the calibration input images and saves the pre/post distortion images
    side-by-side in the specified output directory.
  """
  def visualize_calibration(self):
    # Terminate if the output folder already exists, assume the content is already inside
    calibration_output_images_path = pathlib.Path(self.calibration_output_images_dir)
    if calibration_output_images_path.exists():
      print('{} already exists, not going to regenerate contents'.format(
          str(calibration_output_images_path)))
      return

    # Make the output directory since it doesn't already exists
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
  def get_calibration(self):
    if pathlib.Path(self.camera_calibration_fname).exists():
      self.load_calibration()
    else:
      self.calculate_calibration()
      self.save_calibration()
    self.visualize_calibration()

    return self.calibration


if __name__ == '__main__':
  from pprint import pprint
  cc = CameraCalibrator('camera_calibration/input_images/', 'camera_calibration/output_images/', 9,
      6, 'camera_calibration/camera_calibration.json')
  calibration = cc.get_calibration()
  pprint(calibration)
