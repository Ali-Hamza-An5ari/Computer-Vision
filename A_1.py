# importing the required libraries
import cv2
import numpy as np
import cv2 as cv

# reading image in grayscale
im_target = cv2.imread("img.jpeg", cv2.IMREAD_GRAYSCALE)

im_src = cv.imread('sibalogo.jpg')
pts_src = np.array(im_src)

cap = cv2.VideoCapture("input_video.mp4")


MIN_MATCH_COUNT= 20

# creating the SIFT algorithm
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp_image, desc_image = sift.detectAndCompute(im_target, None)

# initializing the dictionary
index_params = dict(algorithm=0, trees=5)
search_params = dict()

# by using Flann Matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open video")


frame_width = int(cap.get(3))*2
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
result = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

while True:
    ret, frame = cap.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray', grayframe)

    # find the keypoints and descriptors with SIFT
    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)

    # finding nearest match with KNN algorithm
    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)

    # initialize list to keep track of only good points
    good_points = []

    for m, n in matches:
        # append the points according
        # to distance of descriptors
        if (m.distance < 0.6 * n.distance):
            good_points.append(m)
    #print(good_points)

    if len(good_points) > MIN_MATCH_COUNT:
        # maintaining list of index of descriptors
        # in query descriptors
        query_pts = np.float32([kp_image[m.queryIdx]
                               .pt for m in good_points]).reshape(-1, 1, 2)
        # print(query_pts)

        # maintaining list of index of descriptors
        # in train descriptors
        train_pts = np.float32([kp_grayframe[m.trainIdx]
                               .pt for m in good_points]).reshape(-1, 1, 2)
        #print(train_pts)

        # finding  perspective transformation
        # between two planes
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 1.0)

        # ravel function returns
        # contiguous flattened array
        matches_mask = mask.ravel().tolist()

        # initializing height and width of the image
        h, w = im_target.shape

        # saving all points in pts
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        #print(pts)

        # applying perspective algorithm
        dst = cv.perspectiveTransform(pts, matrix)
        print("dst", dst.shape)
        # using drawing function for the frame
        homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

        pts_dst = np.array(dst)

        im_warp = cv2.warpPerspective(im_src, matrix, (frame.shape[1], frame.shape[0]))
        #cv2.imshow("out img", im_out)

        new_img = np.zeros((im_warp.shape[0], im_warp.shape[1], 3), dtype=np.uint8)

        for i in range(0, im_warp.shape[0]):
            for j in range(0, im_warp.shape[1]):
                if (im_warp[i, j][0] == 0):
                    new_img[i, j] = frame[i, j]
                else:
                    new_img[i, j] = im_warp[i, j]
        #cv2.imshow("output", new_img)

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matches_mask,  # draw only inliers
                           flags=2)

        #img3 = cv2.drawMatches(im_target, kp_image, new_img, kp_grayframe, good_points, None, **draw_params)
        img3 = np.concatenate((frame, new_img), axis=1)
        result.write(img3)
        cv2.imshow("output", img3)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
result.release()
cv2.destroyAllWindows()