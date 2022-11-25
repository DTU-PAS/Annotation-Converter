import numpy as np

from annotation_converter.AnnotationConverter import AnnotationConverter
import os
import cv2
import matplotlib.pyplot as plt
def load_annotation_to_img():
    annotation_file = "/home/ronja/data/l515_imgs/20200806_stengard/annotations.xml"
    img_path = "/home/ronja/data/l515_imgs/20200806_stengard/seq0/imgs/20200806_stengard_rgb_0_1628243852051433086.png"
    img_id = os.path.basename(img_path)

    rumex_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    annotations = AnnotationConverter.read_cvat_by_id(annotation_file, img_id)

    # print polygons
    for polyon_ann in annotations.polygon_list:
        test = polyon_ann.get_polygon_points_as_array()
        cv2.drawContours(rumex_img, [polyon_ann.get_polygon_points_as_array()], 0, (0, 0, 255), 2)

    # print bounding boxes
    for bb in annotations.bb_list:
        c1, c2, width, height = bb.get_x() + bb.get_width() / 2, bb.get_y() + bb.get_height() / 2, bb.get_width(), bb.get_height()
        box = cv2.boxPoints(((c1, c2), (width, height), bb.get_rotation()))

        cv2.drawContours(rumex_img, [box.astype(int)], 0, (0, 255, 255), 20)

    plt.imshow(rumex_img)
    plt.show()


if __name__ == "__main__":
    load_annotation_to_img()
