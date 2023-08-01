# Fingerphoto Detection and Segmentation

This is a simple web application, which is made using streamlit, to detect and segment fingerprints from fingerphotos or fingerselfies.

This is deployed in **Streamlit Community Cloud**.
Check-out the link: [Fingerphoto Detection and Segmentation](https://fingerphotodetectionandsegmentation.streamlit.app/)


**Folder Description:**
`YOLOv8` folder consists of train, valid and test folders.

`YOLOv8/train` folder consists of an images folder, which contains images (fingerphotos) to train the Custom YOLOv8 Model, and a labels folder, which contains object (fingerprint) annotation files for every training image in YOLO 1.1 file format.

`YOLOv8/valid` folder consists of an images folder, which contains images (fingerphotos) to validate the Custom YOLOv8 Model, and a labels folder, which contains object (fingerprint) annotation files for every validation image in YOLO 1.1 file format.

`YOLOv8/test` folder consists of images (fingerphotos) in which objects (fingerprints) have to be identified.

`weights` folder consists of two types of detection models (best_detected and last_detected) and two types of segmentation models (best_segmented and last_segmented).

`temp_regions` is a temporary folder needed to store dynamic files during App's execution.


**File Description:**
`StreamlitApp.py` is the script to execute the Streamlit app.

`requirements.txt` contains the libraries and packages needed to execute the app.

`packages.txt` contains of libgl1 which has to be installed to solve import and dependency issues from the library opencv-python.

`weights/best_detected.pt` is the custom YOLOv8 model, which is trained on YOLOv8/train/images and YOLOv8/train/labels, equipped with the best possible weights obtained during the training.

`weights/last_detected.pt` is the custom YOLOv8 model, which is trained on YOLOv8/train/images and YOLOv8/train/labels, equipped with the weights used for the last epoch during the training.

`weights/best_segmented.pt` is the custom YOLOv8 model, which is trained on YOLOv8/train/images and YOLOv8/train/labels, equipped with the best possible weights obtained during the training.

`weights/last_detected.pt` is the custom YOLOv8 model, which is trained on YOLOv8/train/images and YOLOv8/train/labels, equipped with the weights used for the last epoch during the training.

`YOLOv8/data.yaml` is the semantic file to carry training, validation and testing of Custom YOLOv8 Model.

`YOLOv8/data_seg.yaml` is the semantic file to initialize a Custom YOLOv8 Segmentation Model.


**Process Workflow:**
Prefer Python 3.8 environment besides latest versions.

Install the required dependencies and libraries using `pip install -r requirements.txt`.

(Optional) Install libgen1 using `pip install -r packages.txt`.

Place the training dataset inside `YOLOv8/train/images` folder and corresponding annotation files inside `YOLOv8/train/labels` folder.
Place the validation dataset inside `YOLOv8/valid/images` folder and corresponding annotation files inside `YOLOv8/valid/labels` folder.
Place the testing dataset inside `YOLOv8/test` folder.

Train, Validate and Test the Custom YOLOv8 Detection and Segmentation Models in accordance with the `YOLOv8/data.yaml` and `YOLOv8/data_seg.yaml` files.

Place the obtained best.pt and last.pt files, after training the detection and segmentation models inside the `weights` folder.

Run the application using `streamlit run StreamlitApp.py`.

Note: Keep changing the paths to match your filenames denoted by comments in the files `YOLOv8/data.yaml`, `YOLOv8/data_seg.yaml` and `StreamlitApp.py`.