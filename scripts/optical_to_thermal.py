import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import requests
from pycocotools.coco import COCO
from plantcv import plantcv as pcv
import cv2



pcv.params.debug = "print"


def main():

    coco_annotation_file_path = "/home/vedant/Projects/datasets/coco/anno_coco_format/instances_val2017.json"
    coco_images_file_path = "/home/vedant/Projects/datasets/coco/original/test_image/"
    coco_output_folder = "/home/vedant/Projects/datasets/coco/original/output_test/"
    #Create output directory if unavailable
    if (os.path.isdir(coco_output_folder) == False):
            os.mkdir(coco_output_folder)

    #Interested in 'person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
    #Remember these are 1 based so the coco index is 0 for person while it is 1 here.
    coco_interest_category_ids = [1,16,17,18,19,20,21,22,23,24,25]

    #Load all annotations
    coco_annotation = COCO(annotation_file=coco_annotation_file_path)

    # Category IDs.
    cat_ids = coco_annotation.getCatIds()
    print(f"Number of Unique Categories: {len(cat_ids)}")
    print(f"Number of Interested Categories: {len(coco_interest_category_ids)}")

    # All categories.
    cats = coco_annotation.loadCats(coco_interest_category_ids)
    cat_names = [cat["name"] for cat in cats]
    print("Categories Names:")
    print(cat_names)

    # Category Name -> Category ID.
    for cat_name in cat_names :
        query_name = cat_name
        query_id = coco_annotation.getCatIds(catNms=[query_name])[0]
        print("Category Name -> ID:")
        print(f"Category Name: {query_name}, Category ID: {query_id}")


        # Get the ID of all the images containing the object of the category.
        #img_ids = coco_annotation.getImgIds(catIds=[query_id])
        img_ids = [119038]
        print(f"Number of Images Containing {query_name}: {len(img_ids)}")

        # Pick images one by one.
        for img_id in img_ids:
            img_info = coco_annotation.loadImgs([img_id])[0]
            img_file_name = img_info["file_name"]
            img_file_path = coco_images_file_path + img_file_name
            out_file_path = coco_output_folder + img_file_name
            #img_url = img_info["coco_url"]
            print(
                f"Image ID: {img_id}, File Name: {img_file_name}, Image Path: {img_file_path}"
            )

            # Get all the annotations for the specified image.
            ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
            #print(ann_ids)
            anns = coco_annotation.loadAnns(ann_ids)
            final_mask = None


            pcv_img, pcv_path, pcv_filename = pcv.readimage(filename=img_file_path, mode="native")

            # Convert original image to grayscale if not in grayscale already
            if(len(pcv_img.shape)<3):
                pcv_grayscale_img = pcv_img
            else:
                pcv_grayscale_img = pcv.rgb2gray(rgb_img=pcv_img)
            pcv.print_image(pcv_grayscale_img, coco_output_folder + "grayscale.jpg")

            mask = None
            for ann in anns:
                if ann["iscrowd"] == 1:
                    continue
                #Check if annotations are of our objects of interest here.
                #If not of interest skip this annotation.
                if (ann["category_id"] in coco_interest_category_ids):
                    #print(f"Annotation: {ann}")
                    print("ann:")
                    print(ann)
                    if(len(ann["segmentation"]) > 0):
                        seg = ann["segmentation"][0]
                        #Convert into tuple array for PCV segmentation
                        vertices = []
                        for x, y in zip(seg[::2],seg[1::2]):  # No need to end at -1 because that's the default
                            vertices.append([x,y])

                    # Make a custom polygon ROI
                    roi_contour = pcv.roi.custom(img=pcv_img, vertices=vertices)
                    roi_img = pcv_img
                    cv2.drawContours(roi_img,roi_contour.contours[0],-1,(255,0,0),2.5)
                    pcv.print_image(roi_img, coco_output_folder + "roi.jpg")

                    # Generate segmentation mask from plantcv
                    if mask is None:
                        mask = pcv.roi.roi2mask(pcv_img, roi_contour)
                    else:
                        mask = pcv.logical_or(mask, pcv.roi.roi2mask(pcv_img, roi_contour))
            pcv.print_image(mask, coco_output_folder + "mask.jpg")

            #pcv.params.debug = "print"
            # Use the final mask and the grayscale image to generate thermal equivalent
            inverted_mask = pcv.invert(mask)
            pcv.print_image(inverted_mask, coco_output_folder + "inverted_mask.jpg")
            inverted_image = pcv.invert(pcv_grayscale_img)
            pcv.print_image(inverted_image, coco_output_folder + "inverted_grayscale.jpg")
            background_img = pcv.apply_mask(img=inverted_image, mask=inverted_mask, mask_color='white')
            pcv.print_image(background_img, coco_output_folder + "mask_applied.jpg")
            thermal_image = pcv.gaussian_blur(img=background_img, ksize=(9, 9), sigma_x=0, sigma_y=None)
            #thermal_filename = coco_output_folder + str(img_id) + "_thermal.jpg"
            print(out_file_path)
            pcv.print_image(thermal_image, coco_output_folder + "final_thermal.jpg")
            #pcv.print_image(pcv_img, f"{coco_output_folder}{img_id}_optical.jpg")
        # Pick images one by one.
    #end for cat_name in cat_names
    return


if __name__ == "__main__":

    main()
