# Steps used
A document scanner with OpenCV can be accomplished in the following three simple steps:

    Step 1 : Detect edges.
    Step 2 : Use the edges in the image to find the contour (outline) representing the piece of paper being scanned.
    Step 3 : Apply a perspective transform to obtain the top-down view of the document.
# Usage

```
python docscanner.py --image <path-to-image>
```