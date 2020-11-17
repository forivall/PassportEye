'''
PassportEye::MRZ: Machine-readable zone extraction and parsing.
Image processing for MRZ image region extraction.

Author: Konstantin Tretyakov
License: MIT
'''
import io
import numpy as np
from skimage import transform, morphology, filters, measure
from skimage import io as skimage_io # So as not to clash with builtin io
from ..util.pipeline import Pipeline
from ..util.geometry import RotatedBox


class Loader(object):
    """Loads `file` to `img`."""

    __depends__ = []
    __provides__ = ['img']

    def __init__(self, file, as_gray=True, pdf_aware=True):
        self.file = file
        self.as_gray = as_gray
        self.pdf_aware = pdf_aware

    def _imread(self, file):
        """Proxy to skimage.io.imread with some fixes."""
        # For now, we have to select the imageio plugin to read image from byte stream
        # When ski-image v0.15 is released, imageio will be the default plugin, so this
        # code can be simplified at that time.  See issue report and pull request:
        # https://github.com/scikit-image/scikit-image/issues/2889
        # https://github.com/scikit-image/scikit-image/pull/3126
        img = skimage_io.imread(file, as_gray=self.as_gray, plugin='imageio')
        if img is not None and len(img.shape) != 2:
            # The PIL plugin somewhy fails to load some images
            img = skimage_io.imread(file, as_gray=self.as_gray, plugin='matplotlib')
        return img

    def __call__(self):
        if isinstance(self.file, str):
            if self.pdf_aware and self.file.lower().endswith('.pdf'):
                from ..util.pdf import extract_first_jpeg_in_pdf
                with open(self.file, 'rb') as f:
                    img_data = extract_first_jpeg_in_pdf(f)
                if img_data is None:
                    return None
                return self._imread(img_data)
            else:
                return self._imread(self.file)
        elif isinstance(self.file, (bytes, io.IOBase)):
            return self._imread(self.file)
        return None


class Scaler(object):
    """Scales `image` down to `img_scaled` so that its width is at most 250."""

    __depends__ = ['img']
    __provides__ = ['img_small', 'scale_factor']

    def __init__(self, max_width=250):
        self.max_width = max_width

    def __call__(self, img):
        scale_factor = self.max_width / float(img.shape[1])
        if scale_factor <= 1:
            img_small = transform.rescale(img, scale_factor, mode='constant', multichannel=False, anti_aliasing=True)
        else:
            scale_factor = 1.0
            img_small = img
        return img_small, scale_factor


class BooneTransform(object):
    """Processes `img_small` according to Hans Boone's method
    (http://www.pyimagesearch.com/2015/11/30/detecting-machine-readable-zones-in-passport-images/)
    Outputs a `img_binary` - a result of threshold_otsu(closing(sobel(black_tophat(img_small)))"""

    __depends__ = ['img_small']
    __provides__ = ['img_binary']

    def __init__(self, square_size=5):
        self.square_size = square_size

    def __call__(self, img_small):
        m = morphology.square(self.square_size)
        img_th = morphology.black_tophat(img_small, m)
        img_sob = abs(filters.sobel_v(img_th))
        img_closed = morphology.closing(img_sob, m)
        threshold = filters.threshold_otsu(img_closed)
        return img_closed > threshold


class MRZBoxLocator(object):
    """Extracts putative MRZs as RotatedBox instances from the contours of `img_binary`"""

    __depends__ = ['img_binary']
    __provides__ = ['boxes']

    def __init__(self, max_boxes=4, min_points_in_contour=50, min_area=500, min_box_aspect=5, angle_tol=0.1,
                 lineskip_tol=1.5, box_type='bb'):
        self.max_boxes = max_boxes
        self.min_points_in_contour = min_points_in_contour
        self.min_area = min_area
        self.min_box_aspect = min_box_aspect
        self.angle_tol = angle_tol
        self.lineskip_tol = lineskip_tol
        self.box_type = box_type

    def __call__(self, img_binary):
        cs = measure.find_contours(img_binary, 0.5)

        # Collect contours into RotatedBoxes
        results = []
        for c in cs:
            # Now examine the bounding box. If it is too small, we ignore the contour
            ll, ur = np.min(c, 0), np.max(c, 0)
            wh = ur - ll
            if wh[0] * wh[1] < self.min_area:
                continue

            # Finally, construct the rotatedbox. If its aspect ratio is too small, we ignore it
            rb = RotatedBox.from_points(c, self.box_type)
            if rb.height == 0 or rb.width / rb.height < self.min_box_aspect:
                continue

            # All tests fine, add to the list
            results.append(rb)

        # Next sort and leave only max_boxes largest boxes by area
        results.sort(key=lambda x: -x.area)
        return self._merge_boxes(results[0:self.max_boxes])

    def _are_aligned_angles(self, b1, b2):
        "Are two boxes aligned according to their angle?"
        return abs(b1 - b2) <= self.angle_tol or abs(np.pi - abs(b1 - b2)) <= self.angle_tol

    def _are_nearby_parallel_boxes(self, b1, b2):
        "Are two boxes nearby, parallel, and similar in width?"
        if not self._are_aligned_angles(b1.angle, b2.angle):
            return False
        # Otherwise pick the smaller angle and see whether the two boxes are close according to the "up" direction wrt that angle
        angle = min(b1.angle, b2.angle)
        return abs(np.dot(b1.center - b2.center, [-np.sin(angle), np.cos(angle)])) < self.lineskip_tol * (
            b1.height + b2.height) and (b1.width > 0) and (b2.width > 0) and (0.5 < b1.width / b2.width < 2.0)

    def _merge_any_two_boxes(self, box_list):
        """Given a list of boxes, finds two nearby parallel ones and merges them. Returns false if none found."""
        n = len(box_list)
        for i in range(n):
            for j in range(i + 1, n):
                if self._are_nearby_parallel_boxes(box_list[i], box_list[j]):
                    # Remove the two boxes from the list, add a new one
                    a, b = box_list[i], box_list[j]
                    merged_points = np.vstack([a.points, b.points])
                    merged_box = RotatedBox.from_points(merged_points, self.box_type)
                    if merged_box.width / merged_box.height >= self.min_box_aspect:
                        box_list.remove(a)
                        box_list.remove(b)
                        box_list.append(merged_box)
                        return True
        return False

    def _merge_boxes(self, box_list):
        """Mergest nearby parallel boxes in the given list."""
        while self._merge_any_two_boxes(box_list):
            pass
        return box_list


class ExtractAllBoxes(object):
    """Extract all the images from the boxes, for external OCR processing"""

    __provides__ = ['rois']
    __depends__ = ['boxes', 'img', 'img_small', 'scale_factor']

    def __call__(self, boxes, img, img_small, scale_factor):
        rois = []
        scale = 1.0 / scale_factor

        for box in boxes:
            # If the box's angle is np.pi/2 +- 0.01, we shall round it to np.pi/2:
            # this way image extraction is fast and introduces no distortions.
            # and this may be more important than being perfectly straight
            # similar for 0 angle
            if abs(abs(box.angle) - np.pi / 2) <= 0.01:
                box.angle = np.pi / 2
            if abs(box.angle) <= 0.01:
                box.angle = 0.0

            roi = box.extract_from_image(img, scale)
            rois.append(roi)
        return rois


class ROIPipeline(Pipeline):
    """This is a pipeline that just extracts the ROIs"""

    def __init__(self, file):
        super(ROIPipeline, self).__init__()
        self.version = '1.0'  # In principle we might have different pipelines in use, so possible backward compatibility is an issue
        self.file = file
        self.add_component('loader', Loader(file))
        self.add_component('scaler', Scaler())
        self.add_component('boone', BooneTransform())
        self.add_component('box_locator', MRZBoxLocator())
        self.add_component('extractor', ExtractAllBoxes())

    @property
    def result(self):
        return self['rois']


def extract_rois(file: str):
    """The main interface function to this module, encapsulating the recognition pipeline.
       Given an image filename, runs MRZPipeline on it, returning the parsed MRZ object.

    :param file: A filename or a stream to read the file data from.
    """
    return ROIPipeline(file).result
