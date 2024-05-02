# Face-Recognition

The implemented algorithm utilizes the Haar cascade technique for facial feature detection and subsequently employs the LBPH (Local Binary Patterns Histograms) algorithm for facial recognition based on these features.

The Haar cascade algorithm is employed for object detection in images by comparing local features of the image with predefined patterns. In this context, it is utilized specifically for facial detection.

Following the detection of faces, regions containing facial features are extracted and forwarded to the LBPH algorithm for recognition. LBPH is a facial recognition method that generates histograms of small image patterns.
