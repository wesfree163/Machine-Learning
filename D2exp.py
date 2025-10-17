# import os
import sys
import numpy
import itertools
import threading
from pywinauto import mouse

from skimage.io import imread #type: ignore
from sklearn.svm import SVC #type: ignore
from sklearn.metrics import accuracy_score #type: ignore
from skimage.transform import resize #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from time import sleep as stop
from pywinauto import keyboard



import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define image directory and parameters
# image_dir = 'C:\\Users\\Wes\\Downloads\\DEXP-IMG\\VEX TYPE'
# image_dir = 'C:\\Users\\Wes\\Desktop\\D2Exp'

image_size = (64, 64)


done = False



# ########################################################################
# #here is the animation                                              ####
# def animate():                                                      ####
#     for c in itertools.cycle(['|  ', '/  ', '-  ', '\\  ']):        ####
#         if done:                                                    ####
#             break                                                   ####
#         sys.stdout.write('\rloading ' + c)                          ####
#         sys.stdout.flush()                                          ####
#         stop(0.13)                                                  ####
# t = threading.Thread(target=animate)                                ####
# t.start()                                                           ####
# ########################################################################


# ################################################################################################################################################

image_dir = r"C:\Users\Wes\Desktop\D2Exp"
image_size = (128, 128)

images, labels = [], []

for class_name in os.listdir(image_dir):
    class_path = os.path.join(image_dir, class_name)
    if os.path.isdir(class_path):
        for image_filename in os.listdir(class_path):
            image_path = os.path.join(class_path, image_filename)
            try:
                img = imread(image_path)
                if img is not None:
                    img_resized = resize(img, image_size, anti_aliasing=True)
                    images.append(img_resized.flatten())
                    labels.append(class_name)
            except Exception as e:
                print(f"Skipping {image_path}: {e}")

X = np.array(images)
y = np.array(labels)

print(f"Loaded {len(X)} samples.")

if len(X) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = SVC(kernel='linear', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
else:
    print("No images found. Check directory structure and file types.")

# ################################################################################################################################################


# ########################################################################
#VENDOR - TAP                                                       ####
# for i in range(15):                                                 ####
#     stop(8)                                                         ####
#     mouse.click(button='left', coords=(1732, 170))                  ####
# sys.stdout.write('\rCollection at the postmaster')                  ####
done = True                                                         ####
########################################################################
# #POSTMASTER - HOLD                                                 #####
# for i in range(N):                                                 #####
#     stop(8)                                                        #####
#     mouse.click(button='left', coords=(TOP-LEFT))                 #####
# sys.stdout.write('\rCollection at the postmaster')                 #####
# done = True                                                        #####
########################################################################
# #INVENTORY - HOLD - PRIMARY/ENERGY/HEAVY/ARMOR(5)                  #####
# for i in range(N):                                                 #####
#     stop(8)                                                        #####
#     mouse.click(button='left', coords=(1732, 170))                 #####
# sys.stdout.write('\rCollection at the postmaster')                 #####
# done = True                                                        #####
########################################################################
# #XUR'S INVENTORY - TAP                                             #####
# for i in range(15):                                                #####
#     stop(8)                                                        #####
#     mouse.click(button='left', coords=(1732, 170))                 #####
# sys.stdout.write('\rCollection at the postmaster')                 #####
# done = True                                                        #####
########################################################################