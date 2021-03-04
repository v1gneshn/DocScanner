# Steps used
A document scanner + OCR with OpenCV can be accomplished in the following three simple steps:

    Step 1 : Detect edges.
    Step 2 : Use the edges in the image to find the contour (outline) representing the piece of paper being scanned.
    Step 3 : Apply a perspective transform to obtain the top-down view of the document.
    Step 4 : Use the Tesseract-OCR (Here, 
        pytesserect, a python wrapper for Tesseract-OCR) to apply OCR to the aligned document
# Usage 
NOTE : docscanner.py has a document aligner as well as an OCR.
       testocr.py has a standalone OCR tool to read your input text document.

Some points to be noted while supplying the input image :

      1. Make sure the document and the background are in contrast to some extent.
      2. If the document is already in an aligned format (or one with no background), make sure to tweak the bordersize in docscanner.py for a bug-free output.
      3. If you wish to run OCR alone, testocr.py can be used.
      

## Running the modules

```
python docscanner.py --image <path-to-image>
```
Or if you want to do a standalone OCR testing to your image,

```
python testocr.py --image <path-to-image>
```
P.S. -
 Since every image is different, there cannot be a single best hyperparameter set for the OCR to work ideally.Try tweaking in parameters involving gaussian blurring/thresholding/Erosion-Dilation.