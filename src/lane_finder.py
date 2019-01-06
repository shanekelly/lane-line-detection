import cv2
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from camera_calibrator import CameraCalibrator


class LaneFinder:

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
  @param {str} video_path - A path to the video that should be processed for lane detection.
  @param {bool} debug - True to see additional visualization.
  """
  def __init__(self, video_path, debug):
    self.video_path = video_path
    self.debug = debug # toggle more visualization
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
    destination_points = self.get_rectangle_shaped_points(img.shape, width_pct=1.000,
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

    # Combine the two binary images
    img_thresholded_bin = np.zeros_like(img_gradient_x_bin)
    img_thresholded_bin[(img_s_bin == 1) | (img_gradient_x_bin == 1)] = 1

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
  @param {tuple} left_lane_poly_coeffs- A three-element tuple of the polynomial coefficients that
    fit the left lane line (2nd order, 1st order, constant).
  @param {tuple} right_lane_poly_coeffs - A three-element tuple of the polynomial coefficients that
    fit the right lane line (2nd order, 1st order, constant).
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
  def draw_lane(self, img_unwarped, img_warped, left_lane_poly_coeffs, right_lane_poly_coeffs):
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

    cap = cv2.VideoCapture(self.video_path) # open the video stream
    while cap.isOpened(): # while there is another frame of the video to look at
      ret, img = cap.read()
      if ret != True:
        break

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
      # Highlight the currently detected lane in the original image
      img_lane_detected = self.draw_lane(img_undistorted, img_lane_lines_detected,
          left_lane_poly_coeffs, right_lane_poly_coeffs)
      # Get lane line curvature
      left_lane_curvature_radius, right_lane_curvature_radius = \
          self.get_curvature_radii_from_poly_coeffs(img_lane_lines_detected.shape[0],
          left_lane_poly_coeffs, right_lane_poly_coeffs)

      # Visualize
      if self.debug:
        self.plot_images([
          { 'img': cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'title': 'Original' },
          { 'img': cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB), 'title': 'Undistorted' },
          { 'img': img_thresholded, 'title': 'Thresholded', 'cmap': 'gray' },
          { 'img': img_warped, 'title': 'Warped', 'cmap': 'gray' },
          { 'img': img_lane_lines_detected, 'title': 'Lanes Lines Detected' },
          { 'img': cv2.cvtColor(img_lane_detected, cv2.COLOR_BGR2RGB), 'title': 'Lane Detected' }],
          plot_duration=0.25)
      else:
        cv2.imshow('video', np.hstack((cv2.resize(img_lane_detected, None, fx=0.5, fy=0.5),
            cv2.resize(img_lane_lines_detected, None, fx=0.5, fy=0.5))))

      # Update previous lane line fitted polynomial coefficients to use in next search
      prev_left_lane_poly_coeffs = left_lane_poly_coeffs
      prev_right_lane_poly_coeffs = right_lane_poly_coeffs

      if cv2.waitKey(25) & 0xFF == ord('q'): # press q to exit
        break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
  lf = LaneFinder(video_path='media/test_input_videos/project_video.mp4', debug=False)
  lf.main()
