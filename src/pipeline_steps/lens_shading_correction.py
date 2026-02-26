import cv2
import numpy as np


# TODO (andrei aksionau): if to stick to a class variant, find out how to register it to ISP_REGISTRY
class LensShadingCorrection:
    def __init__(self, lns_maps, img_size) -> None:
        self.img_size = img_size

        # TODO (andrei aksionau): most likely lens shading maps are identical across the burst
        # of images. Thus there is no need to have duplicates
        self.lns_maps = lns_maps
        self.interpolated_lns_maps = [self._interpolate(lns_map) for lns_map in lns_maps]

    def _interpolate(self, lns_map):

        # Step 1. Change CFA for the LNS to the correct one
        # From RGGB --> BGGR
        # TODO (andrei aksionau): CFA conversion should be automatic
        lns_bggr = lns_map[:, :, [3, 2, 1, 0]]

        # Step 2. Interpolate
        height, width = self.img_size
        interpolated_lns_planes = []
        for lns_plane in np.unstack(lns_bggr, axis=-1):
            interpolated_lns_planes.append(
                cv2.resize(lns_plane, dsize=(width // 2, height // 2), interpolation=cv2.INTER_LINEAR_EXACT)
            )

        # TODO (andrei aksionau): what is better: to stack a list or to pre-allocate a ndarray
        # and copy into it? If so, then by how much?
        interpolated_lns_map = np.stack(interpolated_lns_planes, axis=-1, dtype=np.float32)

        return interpolated_lns_map

    def apply_single_image(self, img, lns_map):

        # TODO (andrei aksionau): is this necessary to de-interleave and interleave again?
        img_planes = []
        for idx in range(4):
            row_offset, col_offset = divmod(idx, 2)
            img_planes.append(img[row_offset::2, col_offset::2])

        corrected_planes = []
        for img_plane, lns_gain in zip(img_planes, np.unstack(lns_map, axis=-1)):
            corrected_planes.append(img_plane * lns_gain)

        corrected_img = np.empty_like(img)
        for idx, corrected_plane in enumerate(corrected_planes):
            row_offset, col_offset = divmod(idx, 2)
            corrected_img[row_offset::2, col_offset::2] = corrected_plane

        return corrected_img

    def apply(self, imgs):
        # TODO (andrei aksionau): if no functionality is added, replace with list comprehension
        result = []
        for img, lns_map in zip(imgs, self.interpolated_lns_maps):
            corrected_image = self.apply_single_image(img, lns_map)
            result.append(corrected_image)

        return result
