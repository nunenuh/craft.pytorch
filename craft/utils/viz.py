import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from . import box_utils
from . import viz_utils

sub_title = [
    'Ori Image', 'Ori Region', 'Ori Affinity', 'Ori Reg & Aff',
    'Ori Full BBOX', 'Image Region Overlay', 'Image Affinity Overlay', 'Image RegAff Overlay',
    'Calc Image Word BBOX', 'Calc Image Char BBOX', 'Original Char BBOX', 'Original Word BBOX'
]


def show_with_mask(img, mask, alpha_mask=0.5, figsize=(12, 12)):
    plt.figure(figsize=figsize)
    im1 = plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    im2 = plt.imshow(mask, cmap='jet', alpha=alpha_mask, interpolation='bilinear')
    plt.show()


def show_word_char(image, wbbox, cbbox):
    wboxes = box_utils.get_charword_bbox(wbbox, cbbox)
    img = np.array(image)
    colors, i = [(255, 0, 0), (0, 255, 0), (0, 0, 255)], 0
    for wb in wboxes:
        for box in wb:
            cv.polylines(img, np.array([box]), True, colors[i], 2)
        i += 1
        if i >= 2: i = 0

    plt.figure(figsize=(15, 15))
    plt.imshow(img)


def show_grid(images, sub_title, title, autoclose=False, is_save=False, saved_path=None, saved_format='png',
              saved_dpi=150,
              nrow=3, ncol=4, figsize=(30, 30), title_fontsize=None, subtitle_fontsize=None,
              subplot_adjust=(0.1, 0.1, 0.9, 0.2)):
    assert len(images) == len(sub_title), "len of image and len of sub_title is not match! " \
                                          "Make sure you have same number of image and titles!"
    if title_fontsize is None: title_fontsize = int(figsize[0] * 0.618) + figsize[0]
    if subtitle_fontsize is None: subtitle_fontsize = int(title_fontsize * 0.618)

    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
    fig.suptitle(title, fontsize=title_fontsize)
    fig.tight_layout(pad=1)
    fig.subplots_adjust(wspace=subplot_adjust[0], hspace=subplot_adjust[1],
                        top=subplot_adjust[2], bottom=subplot_adjust[3])

    ilen = len(images)
    ic = 0
    for row in axs:
        for ax in row:
            if ilen > ic:
                ax.imshow(images[ic])
                ax.set_title(sub_title[ic], fontsize=subtitle_fontsize)
            ic = ic + 1

    for row in axs:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])

    if is_save:
        fig.savefig(saved_path, bbox_inches='tight', format=saved_format, dpi=saved_dpi)
        print(f'saved file to {saved_path}')
        plt.close(fig)

    if autoclose:
        plt.close(fig)


def visual_analysis(data, path='out.png', title='Image Data', subplot_title=[], alpha_mask=0.5):
    if len(subplot_title) == 0:
        subplot_title = sub_title
    img, t_img, reg, t_reg, aff, t_aff, charbb, wordbb = data
    charbb_img = viz_utils.draw_rect(img, charbb, color=(255, 0, 0), thick=2)
    wordbb_img = viz_utils.draw_rect(img, wordbb, color=(255, 0, 0), thick=2)
    mixedbb_img = viz_utils.draw_rect(charbb_img, wordbb, color=(255, 0, 0), thick=2)
    calc_wordbb_img = viz_utils.word_bbox_draw_rect(img, reg, aff, color=(0, 0, 255), thick=2)
    calc_charbb_img, charbb_img_roi, _ = viz_utils.find_bbox_and_draw_rect(img, reg, color=(0, 0, 255), thick=2)

    fig, axs = plt.subplots(3, 4, figsize=(30, 30))
    fig.suptitle(title, fontsize=32)
    fig.tight_layout(pad=1)
    fig.subplots_adjust(wspace=0.01, hspace=0.01, top=0.95, bottom=0.35)

    axs[0, 0].imshow(img, cmap='gray', interpolation='nearest')
    axs[0, 0].set_title(subplot_title[0], fontsize=25)

    axs[0, 1].imshow(reg, cmap='jet', interpolation='bilinear')
    axs[0, 1].set_title(subplot_title[1], fontsize=25)

    axs[0, 2].imshow(aff, cmap='jet', interpolation='bilinear')
    axs[0, 2].set_title(subplot_title[2], fontsize=25)

    axs[0, 3].imshow(reg + aff, cmap='jet', interpolation='bilinear')
    axs[0, 3].set_title(subplot_title[3], fontsize=25)

    axs[1, 0].imshow(mixedbb_img, cmap='gray', interpolation='nearest')
    axs[1, 0].set_title(subplot_title[4], fontsize=25)

    axs[1, 1].imshow(img, cmap='gray', interpolation='nearest')
    axs[1, 1].imshow(reg, cmap='jet', alpha=alpha_mask, interpolation='bilinear')
    axs[1, 1].set_title(subplot_title[5], fontsize=25)

    axs[1, 2].imshow(img, cmap='gray', interpolation='nearest')
    axs[1, 2].imshow(aff, cmap='jet', alpha=alpha_mask, interpolation='bilinear')
    axs[1, 2].set_title(subplot_title[6], fontsize=25)

    axs[1, 3].imshow(img, cmap='gray', interpolation='nearest')
    axs[1, 3].imshow(reg + aff, cmap='jet', alpha=alpha_mask, interpolation='bilinear')
    axs[1, 3].set_title(subplot_title[7], fontsize=25)

    axs[2, 0].imshow(calc_wordbb_img, cmap='gray', interpolation='nearest')
    axs[2, 0].set_title(subplot_title[8], fontsize=25)

    axs[2, 1].imshow(calc_charbb_img, cmap='gray', interpolation='nearest')
    axs[2, 1].set_title(subplot_title[9], fontsize=25)

    axs[2, 2].imshow(charbb_img, cmap='gray', interpolation='nearest')
    axs[2, 2].set_title(subplot_title[10], fontsize=25)

    axs[2, 3].imshow(wordbb_img, cmap='gray', interpolation='nearest')
    axs[2, 3].set_title(subplot_title[11], fontsize=25)

    for rx in axs:
        for cx in rx:
            cx.set_xticks([])
            cx.set_yticks([])

    fig.savefig(path, bbox_inches='tight', format='png', dpi=150)
    print(f'saved file to {path}')
