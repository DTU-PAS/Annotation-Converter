from annotation_converter.AnnotationConverter import AnnotationConverter

if __name__ == "__main__":
    mask_folder = "sample_data/*.png"
    ann_file = "sample_data/annotations.xml"
    AnnotationConverter.mask_to_cvat(mask_folder, ann_file, "rumex_leaf")
