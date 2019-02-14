# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:26:32 2016

@author: dahoiv
"""

from os.path import basename
from os.path import splitext
import numpy as np
import sqlite3
import nibabel as nib
from nilearn.image import resample_img

import util


class img_data(object):
    def __init__(self, image_id, data_path, temp_data_path):
        self.image_id = image_id
        self.data_path = data_path
        self.temp_data_path = temp_data_path

        self.pre_processed_filepath = None
        self.processed_filepath = None
        self.init_transform = None
        self.transform = None
        self.fixed_image = -1

        self._img_filepath = None
        self._reg_img_filepath = None
        self._label_filepath = None
        self._reg_label_filepath = None
        self._label_inv_filepath = None
        self._reg_brainmask_filepath = None

    @property
    def img_filepath(self):
        if self._img_filepath is not None:
            return self._img_filepath
        conn = sqlite3.connect(self.db_path)
        conn.text_factory = str
        cursor = conn.execute('''SELECT filepath from Images where id = ? ''', (self.image_id,))
        path = cursor.fetchone()
        if path is None:
            print("Could not find data for " + str(self.image_id))
            self._img_filepath = ""
        else:
            self._img_filepath = self.data_path + path[0]
        cursor.close()
        conn.close()

        return self._img_filepath

    @property
    def reg_img_filepath(self):
        if self._reg_img_filepath is not None:
            return self._reg_img_filepath
        conn = sqlite3.connect(self.db_path)
        conn.text_factory = str
        cursor = conn.execute('''SELECT filepath_reg from Images where id = ? ''', (self.image_id,))
        path = cursor.fetchone()
        if path is None:
            print("Could not find data for " + str(self.image_id))
            self._reg_img_filepath = ""
        else:
            self._reg_img_filepath = self.data_path + path[0]
        cursor.close()
        conn.close()

        return self._reg_img_filepath

    def load_db_transforms(self):
        conn = sqlite3.connect(self.db_path)
        conn.text_factory = str
        cursor = conn.execute('''SELECT transform, fixed_image from Images where id = ? ''', (self.image_id,))
        db_temp = cursor.fetchone()
        if db_temp is None:
            return
        self.transform = []
        for _transform in db_temp[0].split(","):
            self.transform.append(self.data_path + _transform.strip())

        self.fixed_image = db_temp[1]
        cursor.close()
        conn.close()

    @property
    def label_filepath(self):
        if self._label_filepath is not None:
            return self._label_filepath

        conn = sqlite3.connect(self.db_path)
        conn.text_factory = str
        cursor = conn.execute('''SELECT filepath from Labels where image_id = ? ''',
                              (self.image_id,))
        self._label_filepath = self.data_path + cursor.fetchone()[0]
        cursor.close()
        conn.close()

        return self._label_filepath

    @property
    def reg_label_filepath(self):
        if self._reg_label_filepath is not None:
            return self._reg_label_filepath

        conn = sqlite3.connect(self.db_path)
        conn.text_factory = str
        cursor = conn.execute('''SELECT filepath_reg from Labels where image_id = ? ''',
                              (self.image_id,))
        self._reg_label_filepath = self.data_path + cursor.fetchone()[0]
        cursor.close()
        conn.close()

        return self._reg_label_filepath

    @property
    def label_inv_filepath(self):
        if self._label_inv_filepath is not None:
            return self._label_inv_filepath

        # resample volume to 1 mm slices
        target_affine_3x3 = np.eye(3) * 1
        img_3d_affine = resample_img(self.label_filepath, target_affine=target_affine_3x3, interpolation='nearest')

        temp_img = img_3d_affine.get_data()
        temp_img[temp_img > 0] = 1
        temp_img = 1 - temp_img
        result_img = nib.Nifti1Image(temp_img, img_3d_affine.affine, img_3d_affine.header)

        self._label_inv_filepath = self.temp_data_path + splitext(splitext(basename(self.label_filepath))[0])[0] + "maskInv.nii.gz"
        nib.save(result_img, self._label_inv_filepath)

        return self._label_inv_filepath

    @property
    def reg_brainmask_filepath(self):
        if self._reg_brainmask_filepath is not None:
            return self._reg_brainmask_filepath

        conn = sqlite3.connect(self.db_path)
        conn.text_factory = str
        cursor = conn.execute('''SELECT filepath_reg from BrainMasks where image_id = ? ''',
                              (self.image_id,))
        path = cursor.fetchone()
        if path is None:
            print("Could not find custom brain mask for " + str(self.image_id))
            self._reg_brainmask_filepath = ""
        else:
            self._reg_brainmask_filepath = self.data_path + path[0]
        cursor.close()
        conn.close()

        return self._reg_brainmask_filepath

    @property
    def db_path(self):
        return self.data_path + "brainSegmentation.db"

    def set_img_filepath(self, filepath):
        self._img_filepath = filepath

    def get_transforms(self):
        if self.init_transform is None:
            return util.ensure_list(self.transform)
        return [self.transform, self.init_transform]
