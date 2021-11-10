def scale(image):
    # HSV mask bounds 
    ref_lower = (100, 100, 100)
    ref_upper = (140, 255, 200)
    
    # We will be applying gaussian smoothing, then HSV conversion, then opening (Erosian -> Dilation) to preprocess image for targeting
    # Gaussian Operation
    blur_image = cv2.GaussianBlur(image, (11, 11), 0)
    # RGB -> HSV
    hsv = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)
    # Morphology: Opening Image
    # Define Mask for Operation
    mask = cv2.inRange(hsv, ref_lower, ref_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Now noise is gone: find edges of reference object (all edges)
    contours,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # Find coordinates in image to draw the target shape outline
    if (len(contours) > 0):
        target = max(contours, key=cv2.contourArea)
        # (x,y), radius = cv2.minEnclosingCircle(target)
        rect = cv2.minAreaRect(target)
        # center = (int(x),int(y))
        # radius = int(radius)
        ((x,y), (width, height), rotation) = rect
        s = f"x {np.round(x)}, y: {np.round(y)}, width: {np.round(width)}, height: {np.round(height)}, rotation: {np.round(rotation)}"
        box = cv2.boxPoints(rect)
        box = np.int64(box)
        M = cv2.moments(target)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # out = cv2.circle(image, center, radius, (0, 255, 0), 4)
        # cv2.circle(image, center, 5, (255, 0, 255), -1)
        cv2.drawContours(image, [box], 0, (255, 0, 0), 4)
        fin =  cv2.putText(image, s, (25, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (200, 0, 50), 2)
        # cv2.imwrite('testout.jpg', fin)
        # print(center, rect)
    else:
        print("Error with thresholds")

    # Getting pixel -> feet conversion
    # Define Wight and Height of the reference in pixels
    refWidth = 5.0
    refHeight = 6.0
    # Ratio: Feet / Pixel
    widthRatio = refWidth / width
    heightRatio = refHeight / height
    # Dims of image
    rows, cols, ch = image.shape
    sideHeight = heightRatio * rows
    sideWidth = widthRatio * cols

    # Write image for white - black image
    image[image != 0] = 255 # change everything to white where pixel is not black
    cv2.imwrite('bw_image.jpg', image)

    # Car width in feet
    cart_width = 0.427
    # Car width in pixels 
    block_width = math.floor(cart_width / widthRatio)
    # Car height in feet
    cart_height = 0.558
    # Car height in pixels 
    block_height = math.floor(cart_height / heightRatio)
    # Get length and width of image 
    matrix_rows = math.floor(rows / block_height)
    matrix_cols = math.floor(cols / block_width)
    matrix_out = [[0 for x in range(matrix_cols)] for y in range(matrix_rows)] 

    # Need greyscale for comparing pixel values
    im_gray = cv2.imread('./p2_test_image.jpg', cv2.IMREAD_GRAYSCALE)

    # Loop over image and fill output matrix for path plan
    clean_threshold = 50
    start_pixel_row = 0
    for i in range(matrix_rows):
        # Set starting pixel starting points
        start_pixel_col = 0
        for j in range(matrix_cols):
            #print("j = {}, i = {}".format(i, j))
            # Loop over all pixels in block 
            #   if # black pixels greater than threshold then no go - o.w. yes clean -> 255
            zero_count = 0
            for u in range(start_pixel_row, start_pixel_row + block_height):
                for v in range(start_pixel_col, start_pixel_col + block_width):
                    #print("u = {}, v = {}".format(u, v))
                    if im_gray[u, v] == 0:
                        zero_count = zero_count + 1
            if zero_count < clean_threshold:
                matrix_out[i][j] = 255;
            # Update starting pixel values
            # Row value same
            # Col value inc by block width
            start_pixel_col = start_pixel_col + block_width
        start_pixel_row = start_pixel_row + block_height

    # Convert aray to image 
    final_blocks = np.asarray(matrix_out)
    cv2.imwrite('scale_out.jpg', final_blocks)

    # Write to file
    with open('block_file.txt', 'w') as f:
        for item in final_blocks:
            f.write("%s\n" % item)

    return final_blocks
    
if __name__ == '__main__':
    # Imports needed
    import cv2
    import numpy as np
    import math

    image = cv2.imread("./p2_test_image.jpg")
    blocks = scale(image)
    print(blocks)
    # return scale(image)