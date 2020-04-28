#
# MIT Licence.
# Written by Milind Deore <tomdeore@gmail.com>
#
# Image tampering for dataset that gives true images and Annotations.
# Synthetic image tampering is required so that model can be trained for
# tampering detection.
#
# download the SUN2012 dataser for object detection from https://groups.csail.mit.edu/vision/SUN/
# Run following command to generate tampered images:
#
#   $ python copy-move-gen.py  --dset=./SUN2012



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import errno
import xml.etree.ElementTree as ET
import numpy as np
from random import randint
from PIL import Image, ImageDraw
import uuid
import cv2

outdir = "." + os.sep + 'train/'

#
# Create output directory, this is like: 'mkdir -p'
#
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

#
# Find Image Folder
#
def find_images_folder(dset):
    for subdir, dirs, files in os.walk(dset):
        if subdir.find("Images"):
            return subdir + os.sep + "Images"

#
# Find Annotation folder
#
def find_anno_folder(dset):
    for subdir, dirs, files in os.walk(dset):
        if subdir.find("Annotations"):
            return subdir + os.sep + "Annotations"


#
# Find annotation file
#
def find_xml_file(anno_dir, find_file):
    for subdir, dirs, files in os.walk(anno_dir):
        for file in files:
            #filepath = subdir + os.sep + file
            if file == find_file:
                found = subdir + os.sep + file
                return found


#
# XML annotation -- Rectangles
#
def find_annotation_from_rectangles(file):

      xny = np.zeros((2,2), dtype=np.int32)

      with open(file, 'rt') as f:
          try:
              tree = ET.parse(f)
              root = tree.getroot()
          except:
              print('ERROR: XML Parse error')
              return(xny, False)

          bndbox_count = 0
          pick_bndbox = 0

          for obj in root.findall('object'):
              for bndbox in obj.findall('bndbox'):
                  bndbox_count = bndbox_count + 1

          # pick a random rect box
          pick_bndbox = randint(1, bndbox_count)

          bbox_count = 0

          for obj in root.findall('object'):
              for bbox in obj.findall('bndbox'):
                  bbox_count = bbox_count + 1
                  if pick_bndbox == bbox_count:
                      xny[0][0] = int(float(bbox.find('xmin').text))
                      xny[1][0] = int(float(bbox.find('xmax').text))
                      xny[0][1] = int(float(bbox.find('ymin').text))
                      xny[1][1] = int(float(bbox.find('ymax').text))

      return (xny, True)


#
# XML Annotations -- Polygons
#
# A XML file can have multiple annotations and moving all of them would not be required, just pick one!
#
def find_annotation_from_polygons(file):

    with open(file, 'rt') as f:
        try:
            tree = ET.parse(f)
            root = tree.getroot()
        except:
            print('ERROR: XML Parse error')
            xny_dummy = np.zeros((2,2), dtype=np.int32)
            return(xny_dummy, False)

        poly_count = 0
        xny_count = 0
        xny_atmost_count = 0
        pick_poly = 0
        points = dict()
        status = False
        p_count = 0
        idx = 0
        idy = 0

        for obj in root.findall('object'):
            for poly in obj.findall('polygon'):
                poly_count = poly_count + 1
                for pts in poly.findall('pt'):
                    for xs in pts.findall('x'):
                        xny_count = xny_count + 1
                if xny_atmost_count < xny_count:
                    xny_atmost_count = xny_count
                points[poly_count] = xny_count
                xny_count = 0

        xny = np.zeros((xny_atmost_count,2), dtype=np.int32)

        # pick a random polygon
        for i in range(poly_count):
            pick_poly = randint(1, poly_count)
            #print('Pick poly %d' % pick_poly)
            if points[pick_poly] > 20:
                status = True
                break


        if status == False:
            return (xny, status)


        for obj in root.findall('object'):
            for poly in obj.findall('polygon'):
                p_count = p_count + 1
                if pick_poly == p_count:
                    for pts in poly.findall('pt'):
                        for xs in pts.findall('x'):
                            if idx > (xny_atmost_count - 1):
                                print('Too large a polygon %d' % idx)
                                break
                            x = int(float(xs.text))
                            xny[idx][0] = x
                            idx = idx + 1
                        for ys in pts.findall('y'):
                            if idy > (xny_atmost_count - 1):
                                print('Too large a polygon %d' % idy)
                                break
                            y = int(float(ys.text))
                            xny[idy][1] = y
                            idy = idy + 1

    return (xny, True)


#
# Position where mask is going to get based.
# Directions: 0:Top, 1:Bottom, 2:Left, 3:Right, 4:DiagonalBottomRight
#             5:DiagonalTopleft, 6:DiagonalTopRight, 7:DiagnoalBottomLeft
#
def move_position(mn, mx, direction):
    x_pos = mx[0] - mn[0]
    x_neg = mn[0] - mx[0]
    y_pos = mx[1] - mn[1]
    y_neg = mn[1] - mx[1]

    if direction == 0:
        # Top
        return [0, y_neg]
    elif direction == 1:
        # Bottom
        return [0, y_pos]
    elif direction == 2:
        # Left
        return [x_neg, 0]
    elif direction == 3:
        # Right
        return [x_pos, 0]
    elif direction == 4:
        # Diagonal - bottom right
        return [x_pos, y_pos]
    elif direction == 5:
        # Diagonal - top left
        return [x_neg, y_neg]
    elif direction == 6:
        # Diagonal - top right
        return [x_pos, y_neg]
    elif direction == 7:
        # Diagonal - bottom left
        return [x_neg, y_pos]
    else:
        print('Invalid direction')


#
# Image tampering happens here
#
def tamper_image(filename, points, anno_type):
    # read image as RGB and add alpha (transparency)
    im = Image.open(filename).convert("RGBA")

    # convert to numpy (for convenience)
    im_array = np.asarray(im)

    # create mask
    mask_im = Image.new('L', (im_array.shape[1], im_array.shape[0]), 0)
    if anno_type == 'rect':
        ImageDraw.Draw(mask_im).rectangle(points, outline=1, fill=1)
    elif anno_type == 'poly':
        ImageDraw.Draw(mask_im).polygon(points, outline=1, fill=1)
    else:
        print('ERROR: Annotation type is missing')
        return False
    mask = np.array(mask_im)

    # assemble new image (uint8: 0-255)
    new_im_array = np.empty(im_array.shape,dtype='uint8')

    # colors (three first columns, RGB)
    new_im_array[:,:,:3] = im_array[:,:,:3]

    # transparency (4th column)
    new_im_array[:,:,3] = mask*255

    # back to Image from numpy
    new_im = Image.fromarray(new_im_array, "RGBA")

    # pasting it back, in random location.
    mx = map(max, zip(*points))
    mn = map(min, zip(*points))

    dir = randint(0, 7)
    position = move_position(mn, mx, dir)
    im.paste(new_im, position, new_im)

    # Save mask with exact same movement.
    tampered_mask = Image.new('L', (im.width, im.height), (0))
    ret, thresh_img = cv2.threshold(mask, 00, 255, cv2.THRESH_BINARY)
    new_im_thresh = Image.fromarray(thresh_img, 'L')
    tampered_mask.paste(new_im_thresh, position)

    # Save Files
    tampered_dir = outdir + 'images/'
    tampered_mask_dir = outdir + 'groundtruth/'
    original_dir = outdir + 'images/'
    original_mask_dir = outdir + 'groundtruth/'
    unique_filename = str(uuid.uuid4())

    # Count non-zeros
    np_tampered_mask = np.array(tampered_mask)
    white_px = np.sum(np_tampered_mask == 255)

    if white_px == 0:
        # if complete black?
        print('INFO: Black image !')
        im.convert('RGB').save(original_dir + os.sep + unique_filename + '.jpg')
        tampered_mask.save(original_mask_dir + os.sep + unique_filename + '.jpg')
    elif white_px < 800:
        print('INFO: Skip this image, too small tampering !')
        # Skip it!
        return False
    else:
        im.convert('RGB').save(tampered_dir + os.sep + unique_filename + '.jpg')
        tampered_mask.save(tampered_mask_dir + os.sep + unique_filename + '.jpg')
        print('INFO: Created.')

    return True


#
# Create output directory with
# Positive and Negative directories
#
def output_mkdir_p():
    # manufactured images
    images_dir = outdir + 'images'
    mkdir_p(images_dir)
    groundtruth_dir = outdir + 'groundtruth'
    mkdir_p(groundtruth_dir)


#
# Main processing loop
#
def main(dset, anno_type):
    #print('Dataset Directory   : %s' % dset)
    #print('Type of annotations : %s' % anno_type)

    output_mkdir_p()

    img_dir = find_images_folder(dset)
    anno_dir = find_anno_folder(dset)

    for subdir, dirs, files in os.walk(img_dir):
        for file in files:
            # Go deep until we find files,
            filepath = subdir + os.sep + file

            if filepath.endswith(".jpg"):
                print('Filename: %s' % filepath)
                find_file = file
                pre, ext = os.path.splitext(find_file)
                find_file = pre + ".xml"

                xfile = find_xml_file(anno_dir, find_file)
                if anno_type == 'rect':
                    xny, retval = find_annotation_from_rectangles(xfile)
                elif anno_type == 'poly':
                    xny, retval = find_annotation_from_polygons(xfile)
                else:
                    print('ERROR: Annotation file type is not specified')
                    return
                if retval == False:
                    break

                print(xny)
                trim_xny = xny[~np.all(xny == 0, axis=1)]
                print(trim_xny)

                for attempt in range(5):
                    retval = tamper_image(filepath, tuple(map(tuple, trim_xny)), anno_type)
                    if retval == True:
                        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = ' -- Copy-Move dataset generator tool -- ')

    parser.add_argument('--dset', required=True,
            help='input root directory for images and annotations. Example: SUN2012 dataset')

    parser.add_argument('--anno_type', required=True,
            help='type of annotation: rectangle(rect) or polygons(poly) Example: --anno_type=rect')

    args = parser.parse_args()
    main(args.dset, args.anno_type)
