from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from matplotlib import pyplot as plt

def thresholding(image):
	
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])
###################################Add border to image###################
row, col = image.shape[:2]
bottom = image[row-2:row, 0:col]
mean = cv2.mean(bottom)[0]

bordersize = 2
image = cv2.copyMakeBorder(
    image,
    top=bordersize,
    bottom=bordersize,
    left=bordersize,
    right=bordersize,
    borderType=cv2.BORDER_CONSTANT,
    value=[mean,mean,mean]
)

##############################################################################
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)
# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gray = cv2.medianBlur(gray, 3)
#gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	print("Approx length :", len(approx))
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply the four point transform to obtain a top-down
# view of the original image

warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio) 

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect

warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#warped = thresholding(gray)

# Apply dilation and erosion to remove some noise
#kernel = np.ones((1, 1), np.uint8)

 #warped = cv2.dilate(warped, kernel, iterations=2)
 #warped = cv2.erode(warped, kernel, iterations=2)
#################

####################
#warped = cv2.adaptiveThreshold(cv2.medianBlur(warped, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
#warped = cv2.medianBlur(warped, 5)


T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
###############   OCR Step   ################

import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
custom_config = r'--oem 3 --psm 6'
text=pytesseract.image_to_string(warped, config=custom_config)
print("#### Step 4: OCR  ######")
print(text)
with open("Output.txt", "w",5 ,"utf-8") as text_file:
 text_file.write(text)
# Plot word boxes on image using pytesseract.image_to_data() function
d = pytesseract.image_to_data(warped, output_type=Output.DICT)
print('DATA KEYS: \n', d.keys())
n_boxes = len(d['text'])
for i in range(n_boxes):
    # condition to only pick boxes with a confidence > 60%
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        image = cv2.rectangle(warped, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(16,12))
plt.imshow(image)
plt.title('SAMPLE INVOICE WITH WORD LEVEL BOXES')
plt.show()