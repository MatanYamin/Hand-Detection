import cv2
import numpy as np
from functools import reduce  # using once in "sort_clockwise" (approved)
import operator as op  # using once in "sort_clockwise" (approved)
import os  # approved for reading all images from folder "images" (approved)
from xlwt import Workbook  # creating an excel file (approved)
from matplotlib import pyplot as plt  # for show plot figure (approved)
# creating an excel file
WB = Workbook()
# name of excel file in python
POINTS = WB.add_sheet('SHEET1')
# setting title "POINT"
POINTS.write(0, 0, 'POINT')
# adding 1-9 numbers
for i in range(1, 10):
    POINTS.write(i, 0, str(i))


def read_image():
    """This function reads all images from 'images' folder.
    doesn't matter what is the name of the images."""
    images = []
    for hand in os.listdir('images'):
        img = cv2.imread(os.path.join('images', hand))
        if img is not None:
            images.append(img)
    return images


def show_image(img, title):
    """this func will recieve image and show it. That is all.
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


def show_plot(img, title):
    """gets an image and title and diplay it on a plot figure as requested"""
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Hand Number: " + title)
    plt.show()


def sort_clockwise(coordinates):
    """this function sort a list with tuples to clockwise order
       for later use of drawing lines"""
    center = tuple(map(op.truediv, reduce(lambda x_, y_: map(op.add, x_, y_), coordinates), [len(coordinates)] * 2))
    coordinates = sorted(coordinates, key=lambda coord: (-135 - np.degrees(
        np.arctan2(*tuple(map(op.sub, center, coord))[::-1]))) % 360)
    return coordinates


def find_dots(img):
    """This function finds the dots coordinates and returns an array of them"""
    # will hold all points
    coordinates = []
    # will hold only relevant points
    points = []
    # losing the side
    img[:, 475:] = 0
    # using for finding the best corners in edged image 65
    corners = cv2.goodFeaturesToTrack(img, 75, 0.085, 61)
    corners = np.int0(corners)
    for corner in corners:
        x, y = corner.ravel()
        if y > 350 or y < 10:  # avoid from top and bottom
            continue
        coordinates.append((x, y))
    # sort in order to start from right to left
    sort_coordinates = sorted(coordinates)
    num_of_dot = 1
    for i in reversed(sort_coordinates):
        # when its 9, break
        if num_of_dot > 9:
            break
        points.append((i[0], i[1]))
        num_of_dot += 1
    return points


def action(hand_images):
    """This function gets an array of all the images and start process them
       inorder to find the specific coordinates for the project"""
    hands = []  # will hold the results of the images
    hand_num = 0  # helps to number the images
    for img in hand_images:
        # the coordinates numbered by order
        order_dots = [3, 4, 5, 6, 7, 8, 9, 1, 2]
        hand_num += 1
        # adding the hand number to the sheet
        POINTS.write(0, hand_num, 'Hand Number: ' + str(hand_num))
        temp = np.copy(img)
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blur the image with 3X3 kernel
        blur = cv2.blur(gray, (3, 3))
        # detecting edges with Canny with 56, 220 (the most accurate I found)
        blur = cv2.Canny(blur, 56, 220)
        # will hold all the coordinates from "find dots"
        coordinates = find_dots(blur)
        # return the coordinates clockwise
        clock_wise = sort_clockwise(list(coordinates))
        first = clock_wise[0]
        before = clock_wise[0]
        # going through the 9 coordinates, mark them, draw line and save to excel
        for coordinate in clock_wise:
            x, y = coordinate[0], coordinate[1]
            # draw the dots on image
            cv2.circle(temp, (x, y), 3, (0, 255, 255), 5)
            # drawing line between
            cv2.line(temp, before, (x, y), (0, 0, 255), 3)
            # hold the last coordinate
            before = (x, y)
            # write to the excel file
            POINTS.write(order_dots.pop(), hand_num, ' X =  ' + str(x) + ' Y = ' + str(y))
        # the last line between the lost dot to the first
        cv2.line(temp, (x, y), first, (0, 0, 255), 3)
        hands.append(temp)
    return hands


def start_program():
    """step 1: read all images from folder
       step 2: go to action() inorder to mark the wanted dots as much as I can"""
    # read all images from "images"
    hands_images = read_image()
    # get all results into an array
    results = action(hands_images)
    # in case we didnt find any dot on any image
    if len(hands_images) == 0:
        print("No dots!")
        return None
    # show every image in plot
    hand_number = 1
    for img in results:
        # function that shows image on a plot figure
        show_plot(img, str(hand_number))
        hand_number += 1
    # saving the excel file in the local folder
    WB.save('Coordinates - projectIM2021_q2.xls')
    # if there are 4 images, show them combined
    if len(hands_images) == 4:
        show_four_images(results[0], results[1], results[2], results[3], "Final Results - Palm: Matan Yamin")


if __name__ == '__main__':
    if not start_program():
        print("No dots found!")
    exit()
