import pytesseract
from pytesseract import Output
from skimage.filters import threshold_local
from matplotlib import pyplot as plt
import cv2
import argparse

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])



#image = cv2.imread("doc_imgs/book2.jpg")
#image = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
gray = get_grayscale(image)    
warped = thresholding(gray)
warped = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255
cv2.imshow("Warped image",warped)
cv2.waitKey(0)
cv2.destroyAllWindows()



pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
custom_config = r'--oem 3 --psm 6'
text=pytesseract.image_to_string(warped, config=custom_config)
print(text)
with open("output_imgs/Output.txt", "w",5 ,"utf-8") as text_file:
 text_file.write(text)
# Plot word boxes on image using pytesseract.image_to_data() function
d = pytesseract.image_to_data(image, output_type=Output.DICT)
print('DATA KEYS: \n', d.keys())
n_boxes = len(d['text'])
for i in range(n_boxes):
    # condition to only pick boxes with a confidence > 60%
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        image = cv2.rectangle(warped, (x, y), (x + w, y + h), (0, 255, 0), 2)

#b,g,r = cv2.split(image)
#rgb_img = cv2.merge([r,g,b])
plt.figure(figsize=(16,12))
plt.imshow(image)
plt.title('SAMPLE WITH WORD LEVEL BOXES')
plt.show()