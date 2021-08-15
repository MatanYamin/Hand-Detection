# Final Project - Image Processing: Question 1
# Matan Yamin, ID: 311544407
import cv2
import numpy as np
import os  # approved from Assaf for read all images from folder "images"
from matplotlib import pyplot as plt  # for show in plot figure


def read_image():
    """This function reads all images that are in 'images' folder.
    doesn't matter what is the name of the images."""
    images = []
    for hand in os.listdir('images'):
        img = cv2.imread(os.path.join('images', hand))
        if img is not None:
            images.append(img)
    return images


def show_image(img, title):
    """this func will get image and show it. that is all.
    press esc to continue"""
    cv2.imshow(title, img)  # show pic
    k = cv2.waitKey(0)
    if k == 27:  # wait until esc
        cv2.destroyAllWindows()


def show_four_images(img1, img2, img3, img4, title):
    """this func helps me show all 4 images together"""
    shape = (460, 250)
    # Get all images in same size for better display
    img1 = cv2.resize(img1, shape)
    img2 = cv2.resize(img2, shape)
    img3 = cv2.resize(img3, shape)
    img4 = cv2.resize(img4, shape)
    # combined 2 images horizontally
    numpy_horizontal1 = np.hstack((img1, img2))
    # combined the rest 2 images horizontally
    numpy_horizontal2 = np.hstack((img3, img4))
    # now combined all vertically to 1 image and display
    numpy_vertical = np.vstack((numpy_horizontal1, numpy_horizontal2))
    # final thing - show the output:
    show_image(numpy_vertical, title)


def action(hand_images):
    all_images = []
    blue = (255, 0, 0)  # color for contours
    red = (0, 0, 255)
    for hand in hand_images:
        # convert the image to grayscale
        gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
        # Applying Canny for edge detection and contours - played with values
        canny = cv2.Canny(gray, 133, 252)
        # find contours for finding convexHull for later
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        marked_hand = np.copy(hand)
        drawing = np.zeros((canny.shape[0], canny.shape[1], 3), np.uint8)
        for i in range(len(contours)):
            cv2.drawContours(drawing, contours, i, blue, 3, 8, hierarchy)
        # convert to gray in order to use "houghCircles"
        gray = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
        # finding all circles with custom parameters for fingertips, works almost for every hand
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, 10, param1=100, param2=32, minRadius=14, maxRadius=21)
        # if there are no circles at all:
        if circles is None:
            continue
        circles = np.uint16(np.around(circles))
        # going through the circles and mark the center
        for circle in circles[0, :]:
            cv2.circle(marked_hand, (circle[0], circle[1]), 2, red, 7)
        all_images.append(marked_hand)
    return all_images


def show_plot(img, title):
    """gets an image and title and diplay it on a plot figure as requested"""
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Hand Number: " + title)
    plt.show()


def start_program():
    """step 1: read all images from folder
       step 2: go to action() inorder to mark the finger tips
       step 3: display all 4 images with a custom function that shows them"""
    hands_images = read_image()  # read a 4 images from folder
    results = action(hands_images)  # get all results into an array
    # in case we didnt find any circle on any image
    if len(hands_images) == 0:
        print("No luck!")
        return -1
    # show every image in plot
    hand_number = 1
    for img in results:
        show_plot(img, str(hand_number))
        hand_number += 1
    # show all 4 images after display with plot
    if len(hands_images) == 4:
        show_four_images(results[0], results[1], results[2], results[3], "Final Results - FingerTips: Matan Yamin")


if __name__ == '__main__':
    start_program()
    exit()
