from pathlib import Path

import cv2
import imutils
import numpy as np
from deskew import determine_skew
from skimage import io
from skimage.filters import threshold_local
from skimage.transform import rotate

# from craft_iqra.utils import box_utils


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def screen_contour_pad(screenCnt, pad_factor=0.05):
    coord = np.copy(screenCnt.reshape(4, 2))

    tl, tr, br, bl = 0, 1, 2, 3
    x_idx, y_idx = 0, 1

    xmin, xmax = min(coord[:, 0]), max(coord[:, 0])
    ymin, ymax = min(coord[:, 1]), max(coord[:, 1])
    w, h = xmax - xmin, ymax - ymin
    pad = max(w, h) * pad_factor

    wpad, hpad = w * pad_factor, h * pad_factor

    coord[tl][x_idx] = coord[tl][x_idx] - wpad
    coord[tl][y_idx] = coord[tl][y_idx] - hpad

    coord[tr][x_idx] = coord[tr][x_idx] - wpad
    coord[tr][y_idx] = coord[tr][y_idx] + hpad

    coord[br][x_idx] = coord[br][x_idx] + wpad
    coord[br][y_idx] = coord[br][y_idx] + hpad

    coord[bl][x_idx] = coord[bl][x_idx] + wpad
    coord[bl][y_idx] = coord[bl][y_idx] - hpad

    coord = coord.reshape(4, 1, 2)

    return coord


def autocrop_warped_image(img, pad_factor=0.02):
    image = img.copy()
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = np.array([])

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = screen_contour_pad(approx, pad_factor=pad_factor)
            break

    if screenCnt.size != 0:
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped = cv2.bitwise_not(warped)
        T = threshold_local(warped, 11, offset=10, method="gaussian")
        return (warped * 255).astype("uint8")
    else:
        return np.array([])


def rotate_image(image, resize=True):
    if len(image.shape)>2:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image
    angle = determine_skew(img_gray)
    rotated = rotate(image, angle, resize=resize)
    return angle, rotated


def get_text_area_coord(boxes, use_pad=True, pad_factor=0.2):
    abox = box_utils.batch_box_coordinate_to_xyminmax(boxes)
    xmin, ymin = np.min(abox[:, 0]), np.min(abox[:, 1]),
    xmax, ymax = np.max(abox[:, 2]), np.max(abox[:, 3])
    box = xmin, ymin, xmax, ymax
    if use_pad:
        box = box_utils.box_pad(box, factor=pad_factor)
    return box


def warp_perspective_by_boxes_area(img, boxes, pad=10, pad_factor=0.1):
    image_warp = img.copy()
    image_warp = cv2.cvtColor(image_warp, cv2.COLOR_BGR2GRAY)
    row, col = image_warp.shape[:2]

    xymm = get_text_area_coord(boxes, use_pad=True, pad_factor=pad_factor)
    coord = box_utils.box_xyminmax_to_coordinates(xymm)
    tl, tr, br, bl = coord
    xmin, ymin, xmax, ymax = xymm
    w, h = xmax - xmin, ymax - ymin
    w, h = int(w + pad), int(h + pad)
    point1 = np.float32([tl, tr, bl, br])
    point2 = np.float32([[10, 10], [w, 10], [10, h], [w, h]])

    P = cv2.getPerspectiveTransform(point1, point2)

    out_perspective = cv2.warpPerspective(image_warp, P, (w, h))
    return out_perspective


def warp_perspective(image, point, size=()):
    img = image.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if len(size) == 2:
        h, w = size
    else:
        h, w = img.shape[:2]
    spoint = np.float32(point)
    tpoint = np.float32([[10, 10], [w, 10], [w, h], [10, h]])
    P = cv2.getPerspectiveTransform(spoint, tpoint)
    return cv2.warpPerspective(img, P, (w, h))


def do_warp_perspective_deskew(image_path, boxes, rotate=True, autosaved=True, pad=100, pad_factor=0.05):
    path = Path(image_path)
    image = cv2.imread(image_path)
    warped = warp_perspective_by_boxes_area(image, boxes, pad=pad, pad_factor=pad_factor)
    if warped.size != 0:
        if rotate:
            angle, rotated = rotate_image(warped)
            rotated = (rotated * 255).astype('uint8')
            print(f'angle : {angle}')
        else:
            rotated = warped

        if autosaved:
            npath = path.parent.joinpath(path.stem + '_deskew' + path.suffix)
            if npath.exists():
                npath.unlink()
            io.imsave(str(npath), rotated)
            # print(npath)
            return str(npath)
        else:
            return angle, rotated
    else:
        # print("warped is failed")
        raise Exception("Image is failed to warped, make sure your image has contrast between object and background!")


def do_autocrop_deskew(image_path, autosaved=True):
    path = Path(image_path)
    image = cv2.imread(image_path)
    warped = autocrop_warped_image(image)
    if warped.size != 0:
        angle, rotated = rotate_image(warped)
        rotated = (rotated * 255).astype('uint8')
        if autosaved:
            npath = path.parent.joinpath(path.stem + '_deskew' + path.suffix)
            if npath.exists():
                npath.unlink()
            io.imsave(str(npath), rotated)
            # print(npath)
            return str(npath)
        else:
            return angle, rotated
    else:
        # print("warped is failed")
        raise Exception("Image is failed to warped, make sure your image has contrast between object and background!")
