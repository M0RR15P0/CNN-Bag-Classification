# Import the libraries and packages
import os, sys
import cv2
import numpy as np

# Define the Drag and Crop function
def mouse_crop(event, x, y, flags, param):
    # Grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping, count
 
    # If the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
 
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
 
    # If the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # Record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # Cropping is finished
 
        refPoint = [(x_start, y_start), (x_end, y_end)]
 
        if len(refPoint) == 2: # When two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped", roi)
            print("Cropped photo saved as: "+os.path.join(dirname[:-1]+"single_"+dirname[-1:],filename[:-4]+"_"+str(count+1)+".jpg"))
            # Save the Cropped Photo
            cv2.imwrite(os.path.join(dirname[:-1]+"single_"+dirname[-1:],filename[:-4]+"_"+str(count+1)+".jpg"),roi)

# List all file names in the directory and loop on the list for cropping
for dirname, dirnames, filenames in os.walk('.'):    
    for subdirname in dirnames:
        subdirectories = os.path.join(dirname, subdirname)
    for filename in filenames:
        cropping = False
        # Only keep file names with .jpg
        if not (filename.endswith(".jpg")):
            break;
        # Show the path of photo to crop
        print("Photo to Crop: "+os.path.join(dirname, filename))
        # Cropping
        count = 0
        while (count >= 0):
            x_start, y_start, x_end, y_end = 0, 0, 0, 0
            image = cv2.imread(os.path.join(dirname, filename));
            oriImage = image.copy()
            cv2.namedWindow("image")
            cv2.setMouseCallback("image", mouse_crop)
            i = image.copy()
            cv2.imshow("image", i)
            k = cv2.waitKey(0)
            if k == 27: # Press Esc to next photo
                cv2.destroyAllWindows();
                break;
            elif k == ord('d'): # Press 'D' to drop the photo if the crop is not good enough
                count = count
                cv2.destroyWindow("Cropped");
                os.remove(os.path.join(dirname[:-1]+"single_"+dirname[-1:],filename[:-4]+"_"+str(count+1)+".jpg"))
                print("Cropped photo deleted.")
            else: # Other keys to continue with cropping
                count = count+1
                cv2.destroyWindow("Cropped");  