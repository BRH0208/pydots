from PIL import Image, ImageDraw
import random
import cv2
import numpy as np
# A simple implementation of cv2 HoughCircles for detecting circles of given radius.
# Originally created to detect panorama's in Google Maps screenshots. 
# BRH0208

# Testing Code: Make an dot-filled image
def create_random_dot_image(image_size, radius, count):

    # Make RGBA image.
    # Note: I am aware I have to convert to BGRA later, but I am lazy and this is test code
    width = image_size[0]
    height = image_size[1]
    image = Image.new("RGBA", (width, height))
    
    
    points_list = [(random.randint(radius, width-radius), random.randint(radius, height-radius)) for x in range(count)]
    # Add points
    image = dot_image(image, points_list, radius)
    return (points_list,image)

# Testing Code: Helper method adds dots to images
def dot_image(image,dots, radius, inner_color = (127,127,127,255),outer_color = (255,255,255,255)):
    draw = ImageDraw.Draw(image)
    for x,y in dots:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
            fill=inner_color, outline=outer_color)
    return image

# Using Cv HoughCircles, Find circles in image
# Returned Parameter
    #Tuple list of circle pixel positions: [(x1,y1),(x2,x2),...]
# Primary Parameters
#   Image: Must be able to be converted to cv2(PIL is ok), image is converted to grayscale w/o transparancy.
#   Radius: The radius of circles to detect
# Hyper Parameters
#   minPadding: Minimum distance between circles(In pixels), it can double-detect circles if this is too high, and may miss overlapping cicles. Radius/2 is default
#   radius_flex: Variance in radius allowed(Measured in pixels). So with radius 5 and flex 1: 4 < radius < 6 is enforced
    # Because you don't expect radius to vary, keep this small, but not zero
#   blur: Should the image be blured prior to preprocessing. Not good for this use case, but better if small obstructions might hamper circle detection
#   sensitivity: param2 of Hough_Gradient detection, determines closeness required for positive circle identification. Lower for more circles, higher for less

def get_circles(image, radius, minPadding = None, radius_flex = 1, blur = False, sensitivity = 5, colorBoundry = (20,200)):
    if(minPadding == None):
        minPadding = radius/2
    # Thank you G4G
    # https://www.geeksforgeeks.org/python/circle-detection-using-opencv-python/
    # Convert to CV2
    image = np.array(image) # NP array
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY) # BGRA image
    
    if(blur): # Blur reduces false positives
        # Is probally not needed here
        image = cv2.blur(image, (3,3))
        
    detected_circles = cv2.HoughCircles(image,
                cv2.HOUGH_GRADIENT, 1,minPadding, param1 = 50,
               param2 = sensitivity, minRadius = radius-radius_flex, maxRadius = radius+radius_flex)
    if detected_circles is None:
        return []
    detected_circles = np.uint16(np.around(detected_circles))[0, :]
    return [(a,b) for a,b,_ in detected_circles if image[b-1,a-1] > colorBoundry[0] and image[b-1,a-1] < colorBoundry[1]]

if __name__ == "__main__":
    # Testing Parameters
    radius = 10
    point_count = 2500
    image_size = (2000,2000)
    true_point_list,image = create_random_dot_image(image_size, radius, point_count)
    # Actual code
    circles = get_circles(image,radius)

    # Calculate "Test score Accuracy"
    # Rated from (0, 1), Asssociates guesses to closest true circles inside radius, losing percentage based on distance from center to edge.
    # Additional Penalty for guesses outside of any radius

    # Get all distances
    p1 = np.array(true_point_list)
    p2 = np.array(circles)
    diff = p1[:, np.newaxis, :] - p2[np.newaxis, :, :]
    precalc_distances = np.sum(diff**2, axis=2)
    points = 0
    banned_true_points = set() # Keep track of circle ID's that have been found already, so can no longer give points
    for i in range(len(circles)):
        distances = {t_id:precalc_distances[t_id,i]
                     for t_id in range(len(true_point_list))
                     if not (t_id in banned_true_points)}

        min_error_id = min(distances,key=distances.get)
        #print(distances[max_error_id],distances[min_error_id])
        if(distances[min_error_id] < radius**2):
            points += 1 - max(0,((distances[min_error_id]-1) / radius**2)) # The max and -1 are to not punish rounding error. You get 1 pixel of error for free
            banned_true_points.add(min_error_id)
        else:
            points -= 1 # Penalty for guesses not in radius
    points = max(0,points / len(true_point_list)) # Scale the score such that it is in range 0-1s
    print("Psudo-Accuracy:",points)
    # Display resulting circles
    dot_image(image,circles,1,inner_color=(255,0,0,255),outer_color=(255,0,0,255))
    image.show()
