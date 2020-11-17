'''
PassportEye::MRZ: Machine-readable zone extraction and parsing.
Pipeline for MRZ extraction using OCR

Author: Konstantin Tretyakov
License: MIT
'''
import numpy as np
from skimage import transform, morphology
from ..util.pipeline import Pipeline
from ..util.ocr import ocr
from .image import Loader, Scaler, BooneTransform, MRZBoxLocator
from .text import MRZ

class FindFirstValidMRZ(object):
    """Iterates over boxes found by MRZBoxLocator, passes them to BoxToMRZ, finds the first valid MRZ
    or the best-scoring MRZ"""

    __provides__ = ['box_idx', 'roi', 'text', 'mrz']
    __depends__ = ['boxes', 'img', 'img_small', 'scale_factor', '__data__']

    def __init__(self, use_original_image=True, extra_cmdline_params=''):
        self.box_to_mrz = BoxToMRZ(use_original_image, extra_cmdline_params=extra_cmdline_params)

    def __call__(self, boxes, img, img_small, scale_factor, data):
        mrzs = []
        data['__debug__mrz'] = []
        for i, b in enumerate(boxes):
            roi, text, mrz = self.box_to_mrz(b, img, img_small, scale_factor)
            data['__debug__mrz'].append((roi, text, mrz))
            if mrz.valid:
                return i, roi, text, mrz
            elif mrz.valid_score > 0:
                mrzs.append((i, roi, text, mrz))
        if not mrzs:
            return None, None, None, None
        else:
            mrzs.sort(key=lambda x: x[3].valid_score)
            return mrzs[-1]


class BoxToMRZ(object):
    """Extracts ROI from the image, corresponding to a box found by MRZBoxLocator, does OCR and MRZ parsing on this region."""

    __provides__ = ['roi', 'text', 'mrz']
    __depends__ = ['box', 'img', 'img_small', 'scale_factor']

    def __init__(self, use_original_image=True, extra_cmdline_params=''):
        """
        :param use_original_image: when True, the ROI is extracted from img, otherwise from img_small
        """
        self.use_original_image = use_original_image
        self.extra_cmdline_params = extra_cmdline_params

    def __call__(self, box, img, img_small, scale_factor):
        img = img if self.use_original_image else img_small
        scale = 1.0 / scale_factor if self.use_original_image else 1.0

        # If the box's angle is np.pi/2 +- 0.01, we shall round it to np.pi/2:
        # this way image extraction is fast and introduces no distortions.
        # and this may be more important than being perfectly straight
        # similar for 0 angle
        if abs(abs(box.angle) - np.pi / 2) <= 0.01:
            box.angle = np.pi / 2
        if abs(box.angle) <= 0.01:
            box.angle = 0.0

        roi = box.extract_from_image(img, scale)
        text = ocr(roi, extra_cmdline_params=self.extra_cmdline_params)

        if '>>' in text or ('>' in text and '<' not in text):
            # Most probably we need to reverse the ROI
            roi = roi[::-1, ::-1]
            text = ocr(roi, extra_cmdline_params=self.extra_cmdline_params)

        if not '<' in text:
            # Assume this is unrecoverable and stop here (TODO: this may be premature, although it saves time on useless stuff)
            return roi, text, MRZ.from_ocr(text)

        mrz = MRZ.from_ocr(text)
        mrz.aux['method'] = 'direct'

        # Now try improving the result via hacks
        if not mrz.valid:
            text, mrz = self._try_larger_image(roi, text, mrz)

        # Sometimes the filter used for enlargement is important!
        if not mrz.valid:
            text, mrz = self._try_larger_image(roi, text, mrz, 1)

        if not mrz.valid:
            text, mrz = self._try_black_tophat(roi, text, mrz)

        return roi, text, mrz

    def _try_larger_image(self, roi, cur_text, cur_mrz, filter_order=3):
        """Attempts to improve the OCR result by scaling the image. If the new mrz is better, returns it, otherwise returns
        the old mrz."""
        if roi.shape[1] <= 700:
            scale_by = int(1050.0 / roi.shape[1] + 0.5)
            roi_lg = transform.rescale(roi, scale_by, order=filter_order, mode='constant', multichannel=False,
                                       anti_aliasing=True)
            new_text = ocr(roi_lg, extra_cmdline_params=self.extra_cmdline_params)
            new_mrz = MRZ.from_ocr(new_text)
            new_mrz.aux['method'] = 'rescaled(%d)' % filter_order
            if new_mrz.valid_score > cur_mrz.valid_score:
                cur_mrz = new_mrz
                cur_text = new_text
        return cur_text, cur_mrz

    def _try_black_tophat(self, roi, cur_text, cur_mrz):
        roi_b = morphology.black_tophat(roi, morphology.disk(5))
        # There are some examples where this line basically hangs for an undetermined amount of time.
        new_text = ocr(roi_b, extra_cmdline_params=self.extra_cmdline_params)
        new_mrz = MRZ.from_ocr(new_text)
        if new_mrz.valid_score > cur_mrz.valid_score:
            new_mrz.aux['method'] = 'black_tophat'
            cur_text, cur_mrz = new_text, new_mrz

        new_text, new_mrz = self._try_larger_image(roi_b, cur_text, cur_mrz)
        if new_mrz.valid_score > cur_mrz.valid_score:
            new_mrz.aux['method'] = 'black_tophat(rescaled(3))'
            cur_text, cur_mrz = new_text, new_mrz

        return cur_text, cur_mrz


class TryOtherMaxWidth(object):
    """
    If mrz was not found so far in the current pipeline,
    changes the max_width parameter of the scaler to 1000 and reruns the pipeline again.
    """

    __provides__ = ['mrz_final']
    __depends__ = ['mrz', '__pipeline__']

    def __init__(self, other_max_width=1000):
        self.other_max_width = other_max_width

    def __call__(self, mrz, __pipeline__):
        # We'll only try this if we see that img_binary.mean() is very small or img.mean() is very large (i.e. image is mostly white).
        if mrz is None and (__pipeline__['img_binary'].mean() < 0.01 or __pipeline__['img'].mean() > 0.95):
            __pipeline__.replace_component('scaler', Scaler(self.other_max_width))
            new_mrz = __pipeline__['mrz']
            if new_mrz is not None:
                new_mrz.aux['method'] = new_mrz.aux['method'] + '|max_width(%d)' % self.other_max_width
            mrz = new_mrz
        return mrz


class MRZPipeline(Pipeline):
    """This is the "currently best-performing" pipeline for parsing MRZ from a given image file."""

    def __init__(self, file, extra_cmdline_params=''):
        super(MRZPipeline, self).__init__()
        self.version = '1.0'  # In principle we might have different pipelines in use, so possible backward compatibility is an issue
        self.file = file
        self.add_component('loader', Loader(file))
        self.add_component('scaler', Scaler())
        self.add_component('boone', BooneTransform())
        self.add_component('box_locator', MRZBoxLocator())
        self.add_component('mrz', FindFirstValidMRZ(extra_cmdline_params=extra_cmdline_params))
        self.add_component('other_max_width', TryOtherMaxWidth())

    @property
    def result(self):
        return self['mrz_final']


def read_mrz(file, save_roi=False, extra_cmdline_params=''):
    """The main interface function to this module, encapsulating the recognition pipeline.
       Given an image filename, runs MRZPipeline on it, returning the parsed MRZ object.

    :param file: A filename or a stream to read the file data from.
    :param save_roi: when this is True, the .aux['roi'] field will contain the Region of Interest where the MRZ was parsed from.
    :param extra_cmdline_params:extra parameters to the ocr.py
    """
    p = MRZPipeline(file, extra_cmdline_params)
    mrz = p.result

    if mrz is not None:
        mrz.aux['text'] = p['text']
        if save_roi:
            mrz.aux['roi'] = p['roi']
    return mrz
