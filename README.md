# Detect-pneumonia-in-medical-images

This work is collaborated with Zhengdi Shen.

## Many object detection methods with deep learning achieved excellent performance to identify objects in im- ages. However, whether they will work to detect visual sig- nals to help diagnose diseases in medical images is still un- der explored. In this project, we explored the deep learning models to detect lung opacity in Chest X-ray images, which is the important signal to diagnose pneumonia. We built a U-net-like stacking model to combine current popular one- stage objection detection methods, YOLO and RetineNet, and the ChexNet, a model designed to classify pneumonia from Chest X-ray images. Our experiments showed this stacking model improved the accuracy to detect lung opac- ity for both of the other two approach separately.

-	Built a U-net-like stacking deep learning model to classify pneumonia regions from Chest X-ray images 
-	Developed 3 U-net architectures to combine coordinates and YOLO and RetineNet, and ChexNet outputs
-	Compared U-net stacking model with YOLOv3, Retinanet on 5000 X-ray images, and U-net model showed highest AU- ROC and average accuracy on detecting pneumonia 


