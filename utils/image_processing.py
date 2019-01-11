import cv2


# Apply the mask img to the frame
def mask(frame, img):
    height, width, channels = frame.shape

    # Resize the img to the size of the frame
    img_resize = cv2.resize(img, (width, height))

    # Transform the img to GRAY scale
    img_resize_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

    # Create the mask from this GRAY image
    ret, masking = cv2.threshold(img_resize_gray, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(masking)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img_resize, img_resize, mask=masking)

    # Put img in ROI and modify the main image
    return cv2.add(img1_bg, img2_fg)
