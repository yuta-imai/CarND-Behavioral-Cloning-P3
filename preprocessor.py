import cv2

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3 

def input_shape():
    return  (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    
def crop(image):
    return image[60:-25, :, :] 

def resize(image):
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def bgr2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

def normalize(image):
    return image/255.0 -0.5

def preprocess(image):
    image = crop(image)
    image = resize(image)
    image = bgr2yuv(image)
    image = normalize(image)
    return image