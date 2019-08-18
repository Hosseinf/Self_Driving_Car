import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_coordinates(image, line_parameters):
	slope, intercept = line_parameters
	y1 = image.shape[0]
	y2 = int(y1*(3/5))
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)

	return np.array([x1,y1,x2,y2])



def average_slope_intercept(image, lines):

	left_fit = []
	right_fit = []
	for line in lines:
		x1, y1, x2, y2 =line.reshape(4)
		# fit polynomial of degree 1 (the last argument)
		parameters = np.polyfit((x1, x2), (y1, y2), 1)

		slope = parameters[0]
		intercept = parameters[1]
		if slope < 0:
			left_fit.append((slope, intercept))
		else:
			right_fit.append((slope, intercept))



	left_fit_average = np.average(left_fit, axis = 0)
	right_fit_average = np.average(right_fit, axis = 0)

	left_line = make_coordinates(image, left_fit_average)
	right_line = make_coordinates(image, right_fit_average)

	return np.array([left_line,right_line])
		# print (right_fit_average)


def canny(lane_image):

	# convert the image to b&w color
	gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5), 0)

	# cv2.Canny(blur,lowthreshold,highThreshold )
	# Filter the image
	canny = cv2.Canny(blur,50,150 )

	return canny


def display(image,lines):
	line_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			# print (line)
			x1,y1,x2,y2 = line.reshape(4)
			# print (x1)
			cv2.line(line_image,(x1,y1), (x2, y2), (255, 0, 0 ), 10)

	return line_image


def region_of_interst(image):
	height = image.shape[0]

	polygons = np.array([[(200, height), (1100, height), (550,250)]])

	# create zeros as the same shape of images corresponding array
	mask = np.zeros_like(image)

	# use openCV fillPoly function to fill the area with polygons  (it takes array of polys)
	cv2.fillPoly(mask, polygons, 255) # third argumrnt is the color of the triangle

	# Apply bitwise to the two images to mask the main image
	masked_image = cv2.bitwise_and(image, mask)

	return masked_image


image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny_image = canny(lane_image)

cropped_image = region_of_interst(canny_image)

# cv2.HoughLines(edges,rho(resolution),theta(resolution),threshold(min number of intersection), place_holder, minLineLength(Minimum length of line. Line segments shorter than this are rejected), maxLineGap (Maximum allowed gap between line segments to treat them as single line) )
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)

# Smoothen the lines
averaged_lines = average_slope_intercept(lane_image, lines)

line_image = display(lane_image, averaged_lines)
# plt.imshow(canny)
# plt.show()

# blend the main image with the line_image
# cv2.addWeighted(image1, wight_for_1st_image, image2, wight_for_2nd_image, gama)
# less weght means the image becomes darker
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

cv2.imshow('result',combo_image)
# display image for specified of milisecond of time. time "0" means it stays for infinite time until we press any button
cv2.waitKey(0)

