import cv2
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import sys

from camera_calibrator import CameraCalibrator


class LaneFinder:
  """
  @param {pathlib.Path} media_path - A Path to some type of media (either a video or image).
  @return {str} - Either 'video' or 'image', based on the filetype extension of `media_path`.
  """
  @staticmethod
  def get_media_type(media_path):
    if media_path.suffix in ('.mp4', '.avi'):
      return 'video'
    elif media_path.suffix in ('.png', '.jpg'):
      return 'image'
    sys.exit('ERROR: input path has an unrecognized extension: {}'.format(media_path.suffix))


  """
  To be used for getting the destination points when warping image perspective.
  @param {tuple} img_size - A tuple with two elements (y size, x size).
  @param {float} top_width_pct - The length of the top side of the road, as a percentage of
    the entire image width.
  @param {float} bot_width_pct - The length of the bottom side of the road, as a percentage
    of the entire image width.
  @param {float} height_pct - The height of the road, as a percentage of the entire image
    height.
  @returns {np.array} - A numpy array of tuples that represent the vertices (x, y) of a
    road-shaped trapezoid.
  """
  @staticmethod
  def get_road_shaped_points(img_size, top_width_pct, bot_width_pct, height_pct):
    mid_x = img_size[1] // 2 # the x index of the middle pixel of the image
    top_width = int(img_size[1] * top_width_pct) # top width, in pixels
    bot_width = int(img_size[1] * bot_width_pct) # bottom width, in pixels
    height = int(img_size[0] * height_pct) # height, in pixels

    # Define the corners of the road-shaped trapezoid
    bot_left = (mid_x - bot_width // 2, img_size[0])
    top_left = (mid_x - top_width // 2, img_size[0] - height)
    bot_right = (mid_x + bot_width // 2, img_size[0])
    top_right = (mid_x + top_width // 2, img_size[0] - height)
    points = np.array([bot_left, top_left, bot_right, top_right]).astype(np.float32)

    return points


  """
  To be used for getting the source points when warping image persepctive.
  @param {tuple} img_size - A tuple with two elements (y size, x size).
  @param {float} width_pct - The width of the rectangle, as a percentage of the entire
    image width.
  @param {float} height_pct - the height of the rectangle, as a percentage of the entire
    image height.
  @returns {np.array} - A numpy array of tuples that represent the vertices (x, y) of a rectangle.
  """
  @staticmethod
  def get_rectangle_shaped_points(img_size, width_pct, height_pct):
    mid_x = img_size[1] // 2 # the x index of the middle pixel of the image
    width = int(img_size[1] * width_pct) # width, in pixels
    height = int(img_size[0] * height_pct) # width, in pixels

    # Define the corners of the rectangle
    bot_left = (mid_x - width // 2, img_size[0])
    top_left = (mid_x - width // 2, img_size[0] - height)
    bot_right = (mid_x + width // 2, img_size[0])
    top_right = (mid_x + width // 2, img_size[0] - height)
    points = np.array([bot_left, top_left, bot_right, top_right]).astype(np.float32)

    return points


  """
  @param {np.array} img - A two-dimensional numpy array of the image to draw on.
  @param {list} points - A list of tuples that represent the vertices (x, y) of the points to draw.
  @param {int} radius - The radius of the points to draw.
  @param {tuple} color - A three element tuple that represents the color of the points to draw.
  """
  @staticmethod
  def draw_points_on_img(img, points, radius, color):
    for point in points:
      cv2.circle(img, (point[0], point[1]), radius, color, -1)


  """
  Constructor for LaneFinder objects. Stores the image filename and camera calibration.
  @param {str} input_media_path - A path to the video that should be processed for lane detection.
  @param {str} [output_media_path=None] - A path to where the output video should be written to. If
    `output_media_path` is set to `None` or is not specified, then no output video will be written.
  @param {bool} [debug=False] - True to visualize additional steps in the lane detection pipeline.
  """
  def __init__(self, input_media_path, output_media_path=None, debug=False):
    self.input_media_path = pathlib.Path(input_media_path)
    if not self.input_media_path.exists(): # exit if the specified `input_media_path` does not exist
      sys.exit('ERROR: video path does not exist: {}'.format(self.input_media_path))

    self.input_media_type = self.get_media_type(self.input_media_path)
    self.output_media_path = pathlib.Path(output_media_path) if output_media_path else None

    self.debug = debug # visualize more steps of the pipeline if set to true

    self.calibration = self.get_calibration() # camera intrinsics for undistortion
    self.meters_per_pixel = { 'x': 3.70 / 700, 'y': 30.0 / 720 } # image unit conversion factors

    self.warping_matrix = None # cache for image perspective warping
    if self.debug:
      self.plot_fig = None # figure for smooth plotting


  """
  Plots an arbitrary number of images specified in `img_dicts` in a grid with three columns and a
    dynamic number of rows.
  @param {list} img_dicts - A list of dictionaries that specify the details of the images to plot.
  @param {np.array} img_dicts[].img - A two-dimensional numpy array image to plot.
  @param {str} img_dicts[].title - The title of the image, to be written above the image.
  @param {str} [img_dicts[].cmap] - The optional color mapping to use to plot the image.
  @param {float} plot_duration - The amount of time in seconds to display each set of images for
    before moving on to the next frame.
  """
  def plot_images(self, img_dicts, plot_duration):
    # Determine the number of rows based on the number of images to be plotted
    num_plots = len(img_dicts)
    num_cols = 3
    num_rows = math.ceil(num_plots / num_cols)
    grid = matplotlib.gridspec.GridSpec(num_rows, num_cols)
    # Re-use the past figure if it exists, so that images can be plotted smoothly like a video
    if self.plot_fig is None:
      self.plot_fig = plt.figure()
      # Make the figure full screen
      mng = plt.get_current_fig_manager()
      mng.resize(*mng.window.maxsize())
    else:
      plt.clf() # clear the figure for

    fontsize = 20 # font size for title text
    for i in range(num_plots):
      ax = self.plot_fig.add_subplot(grid[i])
      img_dict = img_dicts[i]
      if 'cmap' in img_dict: # accept optional `cmap` parameter from `img_dict`
        ax.imshow(img_dict['img'], cmap=img_dict['cmap'])
      else:
        ax.imshow(img_dict['img'])
      ax.set_title(img_dict['title'], fontsize=fontsize)
      ax.set_axis_off() # remove axis tickers

    # Only show the plot for a set amount of time, so that the plot can play like a video
    plt.pause(plot_duration)


  """
  Creates a `CameraCalibrator` object that will use checkerboard images to get camera intrinsics
    used to remove distortion from images.
  @returns {dict} - A dictionary that contains: `camera_matrix`, `distortion_coefficients`,
    `image_shape`, `map_x`, and `map_y`.
  """
  def get_calibration(self):
    cc = CameraCalibrator('camera_calibration/input_images/', 'camera_calibration/output_images/',
        9, 6, 'camera_calibration/camera_calibration.json')

    return cc.get_calibration()


  """
  Removes distortion from an image based on camera intrinsics stored in `self.calibration`.
  @param {np.array} img - A two-dimensional numpy array distorted image to undistort according to
    the camera calibration parameters stored in self.calibration.
  @returns {np.array} - An undistorted version of the input image.
  """
  def undistort(self, img):
    img_undistorted = cv2.remap(img, self.calibration['map_x'], self.calibration['map_y'],
        cv2.INTER_LINEAR)

    return img_undistorted


  """
  @param {np.array} img - A two-dimensional numpy array image to warp.
  @returns {np.arary} - A two-dimensional numpy array that is version of the input image with the
    perspective warped.
  """
  def warp(self, img):
    # Warp the input image such that the vertices of the road-shaped polygon become co-incident with
    #   the vertices of the image.
    source_points = self.get_road_shaped_points(img.shape, top_width_pct=0.100, bot_width_pct=1.000,
        height_pct=0.375)
    destination_points = self.get_rectangle_shaped_points(img.shape, width_pct=0.750,
        height_pct=1.000)
    if self.warping_matrix is None:
      self.warping_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    img_warped = cv2.warpPerspective(img, self.warping_matrix, img.shape[::-1])
    self.draw_points_on_img(img, destination_points, 20, 1)
    self.draw_points_on_img(img, source_points, 10, 1)

    return img_warped


  """
  @param {np.array} img - A two-dimensional numpy array image to unwarp.
  @returns {np.arary} - A two-dimensional numpy array that is version of the input image with the
    perspective warped. This will undo the warping done in `self.warp`.
  """
  def unwarp(self, img_warped):
    img_unwarped = cv2.warpPerspective(img_warped, np.linalg.inv(self.warping_matrix),
        img_warped.shape[:2][::-1])

    return img_unwarped


  """
  @param {np.array} img - A three-channel two-dimensional numpy array image.
  @returns {np.array} - A two-dimensional numpy array binary image that highlights the important
    features of the input image (most notably, the lane lines, but also lots of other stuff that
    will be filtered out later).
  """
  def threshold(self, img):
    # Calculate gradient in the x direction
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gradient_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0) # derivative in x direction
    img_gradient_x = np.absolute(img_gradient_x) # absolute value
    img_gradient_x = np.uint8(img_gradient_x * (255 / np.max(img_gradient_x))) # scale from 0 to 255
    # Threshold x gradient
    gradient_thresh_min = 20
    gradient_thresh_max = 100
    img_gradient_x_bin = np.zeros_like(img_gradient_x)
    img_gradient_x_bin[(img_gradient_x >= gradient_thresh_min) & (img_gradient_x <=
        gradient_thresh_max)] = 1

    # Convert to HLS color space and separate the S channel
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = img_hls[:,:,2]
    # Threshold the saturation color channel
    s_thresh_min = 170
    s_thresh_max = 255
    img_s_bin = np.zeros_like(s_channel)
    img_s_bin[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Convert to LUV color space and separate the L channel
    img_luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = img_luv[:,:,0]
    # Threshold the L color channel
    l_thresh_min = 195
    l_thresh_max = 255
    img_l_bin = np.zeros_like(l_channel)
    img_l_bin[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    # Combine the two binary images
    img_thresholded_bin = np.zeros_like(img_gradient_x_bin)
    img_thresholded_bin[(img_gradient_x_bin == 1) | (img_s_bin == 1) | (img_l_bin == 1)] = 1

    return img_thresholded_bin


  """
  Uses a sliding window method to detect the pixels associated with each of the left and right lane
    lines in an image.
  @param {np.array} img_bin - A two-dimensional numpy array image binary that shows a top-down view
    of a set of left and right lane lines.
  @returns {tuple} - A tuple that contains the following: the x index of every pixel determined to be
    part of the left lane line, the y index of every pixel determined to be part of the left lane
    line, the x index of every pixel determined to be part of the right lane line, the y index of
    every pixel determined to be part of the right lane line, and a two-dimensional numpy array
    image. A modification of `img_bin` with the left lane line drawn in red and the right lane line
    drawn in blue.
  """
  def get_lane_pixels_from_img_bin(self, img_bin):
    img_height, img_width = img_bin.shape # extract image shape information for easier use
    # Create an output image to draw on and visualize
    img_detected = np.dstack((img_bin, img_bin, img_bin))
    # Take a histogram of the bottom half of the image
    num_set_pixels_by_column = np.sum(img_bin[img_height // 2:,:], axis=0)
    # Find the peak of the left and right halves of num_set_pixels_by_column. These will be the
    #   starting point for the left and right lines.
    midpoint = np.int(img_width // 2)
    left_x_start = np.argmax(num_set_pixels_by_column[:midpoint])
    right_x_start = np.argmax(num_set_pixels_by_column[midpoint:]) + midpoint

    num_windows = 9 # number of sliding windows
    win_margin = 100 # the window width will be win_margin * 2
    min_num_pixels = 50 # minimum number of pixels found to recenter window

    # Height of windows based on num_windows and img_height
    window_height = np.int(img_height // num_windows)
    # Identify the x and y indices of all nonzero pixels in the image
    nonzero_y_indices, nonzero_x_indices = img_bin.nonzero()
    # Current positions of lanes to be updated later for each window in num_windows
    curr_left_lane_x = left_x_start
    curr_right_lane_x = right_x_start

    # Create empty lists to receive left and right lane pixel indices
    left_lane_pixel_indices = []
    right_lane_pixel_indices = []

    for window in range(num_windows): # iterate over windows from bottom to top
      # Define the current window boundaries (including the margin in the x direction)
      win_y_min = img_height - (window+1)*window_height
      win_y_max = img_height - window*window_height
      left_lane_win_x_min = curr_left_lane_x - win_margin
      left_lane_win_x_max = curr_left_lane_x + win_margin
      right_lane_win_x_min = curr_right_lane_x - win_margin
      right_lane_win_x_max = curr_right_lane_x + win_margin

      # Draw the current window
      cv2.rectangle(img_detected, (left_lane_win_x_min, win_y_min), (left_lane_win_x_max,
          win_y_max), (0, 255, 0), 2)
      cv2.rectangle(img_detected, (right_lane_win_x_min, win_y_min), (right_lane_win_x_max,
          win_y_max), (0, 255, 0), 2)

      # Identify the nonzero pixels in x and y within the current window
      curr_left_lane_pixel_indices = ((nonzero_y_indices >= win_y_min) &
          (nonzero_y_indices < win_y_max) & (nonzero_x_indices >= left_lane_win_x_min) &
          (nonzero_x_indices < left_lane_win_x_max)).nonzero()[0]
      curr_right_lane_pixel_indices = ((nonzero_y_indices >= win_y_min) &
          (nonzero_y_indices < win_y_max) & (nonzero_x_indices >= right_lane_win_x_min) &
          (nonzero_x_indices < right_lane_win_x_max)).nonzero()[0]

      # Keep track of all lane indices
      left_lane_pixel_indices.append(curr_left_lane_pixel_indices)
      right_lane_pixel_indices.append(curr_right_lane_pixel_indices)

      # If more than min_num_pixels found, recenter next window on their mean position
      if len(curr_left_lane_pixel_indices) > min_num_pixels:
        curr_left_lane_x = np.int(np.mean(nonzero_x_indices[curr_left_lane_pixel_indices]))
      if len(curr_right_lane_pixel_indices) > min_num_pixels:
        curr_right_lane_x = np.int(np.mean(nonzero_x_indices[curr_right_lane_pixel_indices]))

    # Concatenate the arrays of pixel indices (previously was a list of lists of pixels)
    left_lane_pixel_indices = np.concatenate(left_lane_pixel_indices)
    right_lane_pixel_indices = np.concatenate(right_lane_pixel_indices)

    # Extract left lane line and right lane line pixels
    left_lane_pixels_x = nonzero_x_indices[left_lane_pixel_indices]
    left_lane_pixels_y = nonzero_y_indices[left_lane_pixel_indices]
    right_lane_pixels_x = nonzero_x_indices[right_lane_pixel_indices]
    right_lane_pixels_y = nonzero_y_indices[right_lane_pixel_indices]

    # Color pixels that make up each lane
    img_detected[left_lane_pixels_y, left_lane_pixels_x] = (255, 0, 0)
    img_detected[right_lane_pixels_y, right_lane_pixels_x] = (0, 100, 255)

    return (left_lane_pixels_x, left_lane_pixels_y, right_lane_pixels_x,
        right_lane_pixels_y, img_detected)


  """
  Fits a second order polynomial to both the array of left lane line pixels and the array right lane
    line pixels.
  @param {np.array} left_lane_pixels_x - A numpy array that contains the x indices of the pixels
    that make up the left lane.
  @param {np.array} left_lane_pixels_y - A numpy array that contains the y indices of the pixels
    that make up the left lane.
  @param {np.array} right_lane_pixels_x - A numpy array that contains the x indices of the pixels
    that make up the right lane.
  @param {np.array} right_lane_pixels_y - A numpy array that contains the y indices of the pixels
    that make up the right lane.
  @return {tuple} - A tuple that contains: a three-element tuple of the polynomial coefficients
    that fit the left lane line, and a three-element tuple of the polynomial coefficients that fit
    the right lane line.
  """
  def get_poly_coeffs_from_lane_pixels(self, left_lane_pixels_x, left_lane_pixels_y,
      right_lane_pixels_x, right_lane_pixels_y):
    # Fit a second order polynomial to each set of lane pixels
    left_lane_poly_coeffs = np.polyfit(left_lane_pixels_y, left_lane_pixels_x, 2)
    right_lane_poly_coeffs = np.polyfit(right_lane_pixels_y, right_lane_pixels_x, 2)

    return left_lane_poly_coeffs, right_lane_poly_coeffs


  """
  @param {np.array} lane_poly_pixels_y - The input array, each element of which will have an
    associated output value calculated by running the input through the polynomial function.
  @param {tuple} left_lane_poly_coeffs- A three-element tuple of the polynomial coefficients that
    fit the left lane line (2nd order, 1st order, constant).
  @param {tuple} right_lane_poly_coeffs - A three-element tuple of the polynomial coefficients that
    fit the right lane line (2nd order, 1st order, constant).
  @returns {tuple} - A tuple that contains: a numpy array that contains the x indices of the pixels
    that represent the polynomial that was fit to the left lane, a numpy array that contains the x
    indices of the pixels that represent the polynomial that was fit to the right lane, and the y
    indices of the pixels that represent the polynomials that were fit to the lanes.
  """
  def get_poly_pixels_from_poly_coeffs(self, lane_poly_pixels_y, left_lane_poly_coeffs,
      right_lane_poly_coeffs):
    # Discretize polynomial functions
    left_lane_poly_pixels_x = (left_lane_poly_coeffs[0] * lane_poly_pixels_y ** 2 +
        left_lane_poly_coeffs[1] * lane_poly_pixels_y + left_lane_poly_coeffs[2])
    right_lane_poly_pixels_x = (right_lane_poly_coeffs[0] * lane_poly_pixels_y ** 2 +
        right_lane_poly_coeffs[1] * lane_poly_pixels_y + right_lane_poly_coeffs[2])

    return left_lane_poly_pixels_x, right_lane_poly_pixels_x, lane_poly_pixels_y


  """
  Fits a second order polynomial curve to each of the left and right lane lines.
  @param {np.array} img_bin - A two-dimensional numpy array image binary that shows a top-down view
    of a set of left and right lane lines.
  @returns {tuple} - A tuple that contains: a numpy array of pixels (x, y) that make up the
    polynomial that was fit to the left lane line, a numpy array of pixels (x, y) that make up the
    polynomial that was fit to the right lane line, and a two-dimensional numpy array image binary
    with a polynomial line fit and drawn on each of the left and right lane lines.
  """
  def get_poly_coeffs_from_img_bin(self, img_bin):
    # Find the pixels that make up each lane
    (left_lane_pixels_x, left_lane_pixels_y, right_lane_pixels_x,
        right_lane_pixels_y, img_detected) = self.get_lane_pixels_from_img_bin(img_bin)

    # Fit polynomials to the lane lines
    left_lane_poly_coeffs, right_lane_poly_coeffs = \
        self.get_poly_coeffs_from_lane_pixels(left_lane_pixels_x, left_lane_pixels_y,
        right_lane_pixels_x, right_lane_pixels_y)

    # Get the pixels that exist ontop of the fitted polynomials
    left_lane_poly_pixels_x, right_lane_poly_pixels_x, lane_poly_pixels_y = \
        self.get_poly_pixels_from_poly_coeffs(np.linspace(0, img_bin.shape[0]-1, img_bin.shape[0]),
        left_lane_poly_coeffs, right_lane_poly_coeffs)

    # Draw the polynomial lines that are fit to each lane
    left_lane_poly_pixels = np.dstack((left_lane_poly_pixels_x, lane_poly_pixels_y)).astype(int)
    right_lane_poly_pixels = np.dstack((right_lane_poly_pixels_x, lane_poly_pixels_y)).astype(int)
    cv2.polylines(img_detected, [left_lane_poly_pixels, right_lane_poly_pixels], isClosed=False,
        color=(255, 255, 255), thickness=10)

    return left_lane_poly_coeffs, right_lane_poly_coeffs, img_detected


  """
  @param {np.array} img_bin - A two-dimensional numpy array image binary that shows a top-down view
    of a set of left and right lane lines.
  @param {np.array} prev_left_lane_poly_coeffs - A three-element tuple that contains the
    coefficients for the second degree polynomial fitted to the left lane line in the previous
    frame (2nd order, 1st order, constant).
  @param {np.array} prev_right_lane_poly_coeffs - A three-element tuple that contains the
    coefficients for the second degree polynomial fitted to the right lane line in the previous
    frame (2nd order, 1st order, constant).
  """
  def get_poly_coeffs_from_img_bin_and_prev_poly_coeffs(self, img_bin, prev_left_lane_poly_coeffs,
      prev_right_lane_poly_coeffs):
    # Create an output image to draw on and visualize
    img_detected = np.dstack((img_bin, img_bin, img_bin)) * 255

    margin = 100 # margin width around previous polynomail to search for lane line pixels

    # Identify the x and y indices of all nonzero pixels in the image
    nonzero_y_indices, nonzero_x_indices = img_bin.nonzero()

    # Get the pixels that exist ontop of the previously fitted polynomials
    prev_left_lane_poly_pixels_x, prev_right_lane_poly_pixels_x, prev_lane_poly_pixels_y = \
        self.get_poly_pixels_from_poly_coeffs(nonzero_y_indices, prev_left_lane_poly_coeffs,
        prev_right_lane_poly_coeffs)

    # Define the points that make up the edges of the search windows for the current lane lines
    left_lane_win_min_x = prev_left_lane_poly_pixels_x - margin
    left_lane_win_max_x = prev_left_lane_poly_pixels_x + margin
    left_lane_win_min = np.stack((left_lane_win_min_x, prev_lane_poly_pixels_y), axis=1)
    left_lane_win_max = np.flipud(np.stack((left_lane_win_max_x, prev_lane_poly_pixels_y), axis=1))
    left_lane_win = np.vstack((left_lane_win_min, left_lane_win_max))
    right_lane_win_min_x = prev_right_lane_poly_pixels_x - margin
    right_lane_win_max_x = prev_right_lane_poly_pixels_x + margin
    right_lane_win_min = np.stack((right_lane_win_min_x, prev_lane_poly_pixels_y), axis=1)
    right_lane_win_max = np.flipud(np.stack((right_lane_win_max_x, prev_lane_poly_pixels_y),
        axis=1))
    right_lane_win = np.vstack((right_lane_win_min, right_lane_win_max))

    # Draw the search window for finding the current lane lines
    img_win_polygon = np.zeros_like(img_detected)
    cv2.fillPoly(img_win_polygon, np.int32([left_lane_win, right_lane_win]), (0, 255, 0))
    img_detected = cv2.addWeighted(img_detected, 1.0, img_win_polygon, 0.3, 0)

    # Get indices of nonzero pixels within the margin window from the previous fitted polynomials
    left_lane_pixel_indices = ((nonzero_x_indices > left_lane_win_min_x) &
        (nonzero_x_indices < left_lane_win_max_x))
    right_lane_pixel_indices = ((nonzero_x_indices > right_lane_win_min_x) &
        (nonzero_x_indices < right_lane_win_max_x))

    # Extract left lane line and right lane line pixels using nonzero indices
    left_lane_pixels_x = nonzero_x_indices[left_lane_pixel_indices]
    left_lane_pixels_y = nonzero_y_indices[left_lane_pixel_indices]
    right_lane_pixels_x = nonzero_x_indices[right_lane_pixel_indices]
    right_lane_pixels_y = nonzero_y_indices[right_lane_pixel_indices]

    # Draw color in the pixels that make up each lane
    img_detected[left_lane_pixels_y, left_lane_pixels_x] = (255, 0, 0)
    img_detected[right_lane_pixels_y, right_lane_pixels_x] = (0, 100, 255)

    # Get the coefficients of the new fitted polynomials for each lane line
    left_lane_poly_coeffs, right_lane_poly_coeffs = \
        self.get_poly_coeffs_from_lane_pixels(left_lane_pixels_x, left_lane_pixels_y,
        right_lane_pixels_x, right_lane_pixels_y)

    # Get the pixels that make up the newly fitted polynomials
    left_lane_poly_pixels_x, right_lane_poly_pixels_x, lane_poly_pixels_y = \
        self.get_poly_pixels_from_poly_coeffs(np.linspace(0, img_bin.shape[1]-1, img_bin.shape[0]),
        left_lane_poly_coeffs, right_lane_poly_coeffs)

    # Draw the polynomial lines that are fit to each lane
    left_lane_poly_pixels = np.dstack((left_lane_poly_pixels_x, lane_poly_pixels_y)).astype(int)
    right_lane_poly_pixels = np.dstack((right_lane_poly_pixels_x, lane_poly_pixels_y)).astype(int)
    cv2.polylines(img_detected, [left_lane_poly_pixels, right_lane_poly_pixels], isClosed=False,
        color=(255, 255, 255), thickness=10)

    return left_lane_poly_coeffs, right_lane_poly_coeffs, img_detected


  """
  @param {int} img_shape_y - The height of the image that the polynomial coefficients came from.
    This will be the point that the polynomial function is evaluated at for curvature (effectively,
    the bottom of the image).
  @param {tuple} left_lane_poly_coeffs- A three-element tuple of the polynomial coefficients (2nd
    order, 1st order, constant) that fit the left lane line in pixels.
  @param {tuple} right_lane_poly_coeffs- A three-element tuple of the polynomial coefficients (2nd
    order, 1st order, constant) that fit the right lane line in pixels.
  @returns {tuple} - A tuple that contains: the curvature of the left lane line in meters and the
    curvature of the right lane line in meters.
  """
  def get_curvature_radii_from_poly_coeffs(self, img_shape_y, left_lane_poly_coeffs,
      right_lane_poly_coeffs):
    # Some math about converting polynomial coefficients from pixels to meters, where
    #   w=self.meters_per_pixel['x'] and h=self.meters_per_pixel['y'].
    # x/w = (a)*(y/h)^2 + (b)*(y/h) + (c)
    # x/w = (a/h^2)*(y)^2 + (b/h)*(y) + (c)
    # x = (a*w/h^2)*(y)^2 + (b*w/h)*(y) + (c*w)
    left_lane_poly_meter_coeffs = [(left_lane_poly_coeffs[0] * self.meters_per_pixel['x'] /
        self.meters_per_pixel['y'] ** 2), (left_lane_poly_coeffs[1] * self.meters_per_pixel['x'] /
        self.meters_per_pixel['y']), (left_lane_poly_coeffs[2] * self.meters_per_pixel['x'])]
    right_lane_poly_meter_coeffs = [(right_lane_poly_coeffs[0] * self.meters_per_pixel['x'] /
        self.meters_per_pixel['y'] ** 2), (right_lane_poly_coeffs[1] * self.meters_per_pixel['x'] /
        self.meters_per_pixel['y']), (right_lane_poly_coeffs[2] * self.meters_per_pixel['x'])]

        # Calculate lane curvature using equation from
        #   https://www.intmath.com/applications-differentiation/8-radius-curvature.php
    left_lane_curvature_radius = (((1 + (2 * left_lane_poly_meter_coeffs[0] * img_shape_y *
        self.meters_per_pixel['y'] + left_lane_poly_meter_coeffs[1]) ** 2) ** 1.5) /
        np.absolute(2 * left_lane_poly_meter_coeffs[0]))
    right_lane_curvature_radius = (((1 + (2 * right_lane_poly_meter_coeffs[0] * img_shape_y *
        self.meters_per_pixel['y'] + right_lane_poly_meter_coeffs[1]) ** 2) ** 1.5) /
        np.absolute(2 * right_lane_poly_meter_coeffs[0]))

    return left_lane_curvature_radius, right_lane_curvature_radius


  """
  @param {int} img_shape_y - The height of the image that the polynomial coefficients came from.
    This will be the point that the polynomial function is evaluated at for curvature (effectively,
    the bottom of the image).
  @param {tuple} left_lane_poly_coeffs- A three-element tuple of the polynomial coefficients (2nd
    order, 1st order, constant) that fit the left lane line in pixels.
  @param {tuple} right_lane_poly_coeffs- A three-element tuple of the polynomial coefficients (2nd
    order, 1st order, constant) that fit the right lane line in pixels.
  @returns {float} - The relative offset between the car and the center of the detected lane in the
    x direction in meters. A positive offset means the car is too far right.
  """
  def get_car_to_lane_offset(self, img_shape, left_lane_poly_coeffs, right_lane_poly_coeffs):
    # Get the center of the car, which is also the center of the image
    center_of_car_x_pixels = img_shape[1] / 2

    # Get the center of the bottom of the lane
    left_bottom_of_lane_x_pixels = (left_lane_poly_coeffs[0] * img_shape[0] ** 2 +
      left_lane_poly_coeffs[1] * img_shape[0] + left_lane_poly_coeffs[2])
    right_bottom_of_lane_x_pixels = (right_lane_poly_coeffs[0] * img_shape[0] ** 2 +
      right_lane_poly_coeffs[1] * img_shape[0] + right_lane_poly_coeffs[2])
    center_of_lane_x_pixels = (left_bottom_of_lane_x_pixels + right_bottom_of_lane_x_pixels) / 2

    # Get the distance between the car and the lane in meters
    car_to_lane_offset_pixels = center_of_car_x_pixels - center_of_lane_x_pixels
    car_to_lane_offset_meters = car_to_lane_offset_pixels * self.meters_per_pixel['x']

    return car_to_lane_offset_meters


  """
  @param {np.array} img_unwarped - A two-dimensional numpy array image to overlay with the drawn
    lane polygon.
  @param {np.arary} img_warped - A two-dimensional numpy array image used to determine how to draw
    the lane polygon. Should be a warped image with a top-down perspective of the lane.
  @param {tuple} left_lane_poly_coeffs- A three-element tuple of the polynomial coefficients that
    fit the left lane line (2nd order, 1st order, constant).
  @param {tuple} right_lane_poly_coeffs - A three-element tuple of the polynomial coefficients that
    fit the right lane line (2nd order, 1st order, constant).
  @returns {np.array} - A two-dimensional numpy array image with the road polygon overlayed on top
    of `img_unwarped`.
  """
  def draw_lane(self, img_unwarped, img_warped, left_lane_poly_coeffs, right_lane_poly_coeffs,
      left_lane_curvature_radius, right_lane_curvature_radius, car_to_lane_offset):
    # Get the pixels that define the left and right edges of the lane
    left_lane_poly_pixels_x, right_lane_poly_pixels_x, lane_poly_pixels_y = \
        self.get_poly_pixels_from_poly_coeffs(np.linspace(0, img_warped.shape[0] - 1,
        img_warped.shape[0]), left_lane_poly_coeffs, right_lane_poly_coeffs)
    left_lane_poly_pixels = np.stack((left_lane_poly_pixels_x, lane_poly_pixels_y), axis=1)
    right_lane_poly_pixels = np.flipud(np.stack((right_lane_poly_pixels_x, lane_poly_pixels_y),
        axis=1))
    lane_poly_pixels = np.concatenate((left_lane_poly_pixels, right_lane_poly_pixels))

    # Draw a polygon around the lane pixels
    img_lane_polygon_warped = np.zeros_like(img_warped)
    cv2.fillPoly(img_lane_polygon_warped, np.int32([lane_poly_pixels]), (0, 255, 0))
    img_lane_polygon_unwarped = self.unwarp(img_lane_polygon_warped)
    img_lane_detected = cv2.addWeighted(img_unwarped, 1.0, img_lane_polygon_unwarped, 0.3, 0)

    # Calculate the average curvature radius of the two lane lines
    curvature_radius = (left_lane_curvature_radius + right_lane_curvature_radius) / 2

    # Draw the text that displays the radius of curvature and the car to lane offset
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_color = (255, 255, 255)
    font_thickness = 3
    curvature_text = 'Radius of Curvature: {:4.2f}m'.format(curvature_radius)
    lane_offset_text = 'Car to Lane Offset: {:4.2f}m'.format(car_to_lane_offset)
    cv2.putText(img_lane_detected, curvature_text, (25, 50), font_face, font_scale, font_color,
        font_thickness)
    cv2.putText(img_lane_detected, lane_offset_text, (25, 100), font_face, font_scale, font_color,
        font_thickness)

    return img_lane_detected


  """
  The main function of the `LaneFinder` class. Processes the specified video to find the lane and
    visualizes the results.
  """
  def main(self):
    # The polynomials fitted to the lane lines in the previous video frame will be useful in finding
    #   the lane lines in the current video frame.
    prev_left_lane_poly_coeffs = None
    prev_right_lane_poly_coeffs = None

    if self.input_media_type is 'video':
      video_reader = cv2.VideoCapture(str(self.input_media_path)) # open the video stream

    if self.output_media_path is not None:
      if self.input_media_type is 'video':
        video_codec = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(str(self.output_media_path), video_codec, 20.0, (1281 * 2, 721))

    keep_looping = True
    while keep_looping: # while there is another frame of the input media to look at
      if self.input_media_type is 'video':
        ret, img = video_reader.read()
        if ret != True:
          break
      else: # self.input_media_type is 'image'
        img = cv2.imread(str(self.input_media_path))

      # Remove image distortion based on camera intrinsics
      img_undistorted = self.undistort(img)
      # Create a binary image that highlights important features, including the lane lanes
      img_thresholded = self.threshold(img_undistorted)
      # Warp perspective to a top-down view of the lane
      img_warped = self.warp(img_thresholded)
      # Detect which set bits represent the lane lines and fit polynomials to the lane lines
      if prev_left_lane_poly_coeffs is not None and prev_right_lane_poly_coeffs is not None:
        left_lane_poly_coeffs, right_lane_poly_coeffs, img_lane_lines_detected = \
            self.get_poly_coeffs_from_img_bin_and_prev_poly_coeffs(img_warped,
            prev_left_lane_poly_coeffs, prev_right_lane_poly_coeffs)
      else:
        left_lane_poly_coeffs, right_lane_poly_coeffs, img_lane_lines_detected = \
            self.get_poly_coeffs_from_img_bin(img_warped)
      # Get lane line curvature
      left_lane_curvature_radius, right_lane_curvature_radius = \
          self.get_curvature_radii_from_poly_coeffs(img_lane_lines_detected.shape[0],
          left_lane_poly_coeffs, right_lane_poly_coeffs)
      # Get the x axis offset between the car and the detected lane
      car_to_lane_offset = self.get_car_to_lane_offset(img_lane_lines_detected.shape,
          left_lane_poly_coeffs, right_lane_poly_coeffs)
      # Highlight the currently detected lane in the original image
      img_lane_detected = self.draw_lane(img_undistorted, img_lane_lines_detected,
          left_lane_poly_coeffs, right_lane_poly_coeffs, left_lane_curvature_radius,
          right_lane_curvature_radius, car_to_lane_offset)
      # Update previous lane line fitted polynomial coefficients to use in next search
      prev_left_lane_poly_coeffs = left_lane_poly_coeffs
      prev_right_lane_poly_coeffs = right_lane_poly_coeffs

      # Visualize
      if self.debug:
        # Format images so they can all be merged next to each other and displayed
        img = cv2.resize(img, (img.shape[1] + 1, img.shape[0] + 1))
        img_thresholded = np.dstack((img_thresholded, img_thresholded, img_thresholded)) * 255
        img_warped = np.dstack((img_warped, img_warped, img_warped)) * 255
        img_final = np.vstack((np.hstack((img, img_undistorted, img_thresholded)),
          np.hstack((img_warped, img_lane_lines_detected, img_lane_detected))))
      else:
        img_final = np.hstack((img_lane_detected, img_lane_lines_detected))

      cv2.imshow('final_visualization', cv2.resize(img_final, None, fx=0.5, fy=0.5))
      if self.output_media_path is not None:
        if self.input_media_type is 'video':
          video_writer.write(img_final) # write the current visualization frame to a video
        else: # self.input_media_type is 'image'
          cv2.imwrite(str(self.output_media_path), img_final) # write the visualization image

      if self.input_media_type is 'video':
        wait_time_ms = 25 # show each frame for 25 milliseconds
      else: # self.input_media_type is 'image'
        wait_time_ms = 0 # show the image indefinitely
      # Show the frame for the specified time and if the user presses the 'q' key, then exit
      if cv2.waitKey(wait_time_ms) & 0xFF == ord('q'):
        break

      # Whether or not to stay in the while loop
      if self.input_media_type is 'video':
        keep_looping = video_reader.isOpened()
      else: # self.input_media_type is 'image'
        keep_looping = False # an image only has one frame, so we should instantly break from the while loop

    if self.input_media_type is 'video':
      video_reader.release() # close the video reader
      if self.output_media_path is not None:
        video_writer.release() # close the video writer

    cv2.destroyAllWindows() # close the opencv window


if __name__ == '__main__':
  import argparse

  # Get command line arguments
  parser = argparse.ArgumentParser(description='Find the lane in a video stream.')
  parser.add_argument('--input-media-path', required=True,  help='The path to the input media that'
    ' should be processed. This can be either a video or an image. A good option would be'
    ' `media/test_input_videos/project_video.mp4` or `media/test_input_images/straight_lines_1.jpg`'
    '.')
  parser.add_argument('--output-media-path', help='If set, then the final visualization of the'
    ' processed media will be written to the specified path. A good option would be'
    ' `media/test_output_videos/<name>.avi` or `media/test_output_images/<name>.png`.')
  parser.add_argument('--debug', action='store_true', help='Set debug mode. Shows more'
    ' visualizations of the different steps in the lane detection pipeline.')
  args = parser.parse_args()

  # Create a LaneFinder object and run the main function with the command line arguments
  lf = LaneFinder(args.input_media_path, args.output_media_path, args.debug)
  lf.main()
