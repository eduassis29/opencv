import cv2

# Load an image
image = cv2.imread('DogoTchanka.jpg')

# Display the image
cv2.imshow('Image', image)

# Wait for any key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
