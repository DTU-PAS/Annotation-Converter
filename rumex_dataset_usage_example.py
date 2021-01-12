from annotation_converter.AnnotationConverter import AnnotationConverter
import glob
import os
import cv2
import matplotlib.pyplot as plt


def project_mask_on_img(mask, img):
    out = cv2.addWeighted(mask, 1, img, 0.8, 0)
    return out

if __name__ == "__main__":
    #ToDo: Set you custom path here!
    PATH_TO_DATASET = "/home/rog/data/data_iphone6/annotated_rumex"

    annotation_file = f"{PATH_TO_DATASET}/annotations.xml"
    # Load image + mask
    img_files = glob.glob(f"{PATH_TO_DATASET}/imgs/*.jpg")

    for img_file in img_files:
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotation = AnnotationConverter.read_cvat_by_id(annotation_file, os.path.basename(img_file))

        # img + segmentation mask
        seg_mask = AnnotationConverter.get_mask(annotation, ["rumex_leaf"], img.shape[0], img.shape[1], (100, 100, 100))
        plt.imshow(project_mask_on_img(seg_mask, img))
        plt.show()

        # img + bounding boxes
        polygons = annotation.get_polygons()
        for polygon in polygons:
            bb = polygon.to_bounding_box()
            cv2.rectangle(img, (bb.get_x(), bb.get_y()), (bb.get_x() + bb.get_width(), bb.get_y() + bb.get_height()), (255, 0, 0), 10)
        plt.imshow(img)
        plt.show()

