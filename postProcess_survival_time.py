# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:41:50 2016

@author: dahoiv
"""

from openpyxl import Workbook
import nipype.interfaces.semtools.registration.brainsresample as brainsresample
import datetime
import util
import sqlite3
import nibabel as nib
import numpy as np
import os


BRAINSResample_PATH = '/home/leb/dev/BRAINSTools/build/bin/BRAINSResample'

def process(folder):
    """ Post process data """
    print(folder)
    util.setup(folder)

    image_ids, survival_days = util.get_image_id_and_survival_days(study_id="GBM_survival_time")
    result = util.post_calculations(image_ids)
    print(len(result['all']))
    util.avg_calculation(result['all'], 'all_', None, True, folder, save_sum=True)
    util.avg_calculation(result['img'], 'img_', None, True, folder)

    for label in result:
        if label == 'img':
            continue
        util.avg_calculation(result[label], 'survival_time', survival_days, True, folder, default_value=-1)
        util.median_calculation(result[label], 'survival_time', survival_days, True, folder, default_value=-1)

def process_labels(folder, pids_to_exclude=None):
    """ Create Excel-document with overview of which brain areas are affected by the tumors"""
    print(folder)
    util.setup(folder)
    conn = sqlite3.connect(util.DB_PATH, timeout=120)
    conn.text_factory = str
    pids, image_ids = util.get_pids_and_image_ids(study_id="GBM_survival_time",exclude_pid=pids_to_exclude)


    atlas_path = util.ATLAS_FOLDER_PATH + "Hammers/Hammers_mith-n30r95-MaxProbMap-full-MNI152-SPM12.nii.gz"
    resample = brainsresample.BRAINSResample(command=BRAINSResample_PATH,
                                             inputVolume=atlas_path,
                                             outputVolume=os.path.abspath(folder +
                                                                          'Hammers_mith-n30r95-MaxProbMap-full'
                                                                          '-MNI152-SPM12_resample.nii.gz'),
                                             referenceVolume=os.path.abspath(util.TEMPLATE_VOLUME))
    resample.run()

    img = nib.load(folder + 'Hammers_mith-n30r95-MaxProbMap-full-MNI152-SPM12_resample.nii.gz')
    lobes_brain = img.get_data()
    label_defs = util.get_label_defs_hammers_mith()
    res_lobes_brain = {}
    ventricle_label = 49
    com_ventricle, com_idx_ventricle =  util.get_center_of_mass(folder + 'Hammers_mith-n30r95-MaxProbMap-full-MNI152-SPM12_resample.nii.gz',ventricle_label)

    book = Workbook()
    sheet = book.active

    sheet.cell(row=1, column=1).value = 'PID'
    sheet.cell(row=1, column=2).value = 'Lobe, center of tumor'
    sheet.cell(row=1, column=3).value = 'Distance between centres of tumor and 3rd ventricle (mm)'
    i = 3
    label_defs_to_column = {}
    for key in label_defs:
        i += 1
        sheet.cell(row=1, column=i).value = label_defs[key]
        label_defs_to_column[key] = i
    # sheet.cell(row=1, column=3).value = 'Center of mass'
    k = 2
    for (pid, _id) in zip(pids, image_ids):

        _filepath = conn.execute("SELECT filepath_reg from Labels where image_id = ?",
                                 (_id, )).fetchone()[0]
        if _filepath is None:
            print("No filepath for ", pid)
            continue

        com, com_idx = util.get_center_of_mass(util.DATA_FOLDER + _filepath)
        print(pid, com_idx)
        dist_to_ventricle = np.linalg.norm(np.array(com)-np.array(com_ventricle))

        lobe = label_defs.get(lobes_brain[com_idx[0], com_idx[1], com_idx[2]], 'other')
        res_lobes_brain[pid] = lobe

        tumor_data = nib.load(util.DATA_FOLDER + _filepath).get_data()
        union_data = lobes_brain * tumor_data
        union_data = union_data.flatten()
        lobe_overlap = ''
        for column in range(1, 1+max(label_defs_to_column.values())):
            sheet.cell(row=k, column=column).value = 0
        for _lobe in np.unique(union_data):
            column = label_defs_to_column.get(_lobe)
            if column is None:
                continue
            sheet.cell(row=k, column=column).value = 1
            lobe_overlap += label_defs.get(_lobe, '') + ', '

        sheet.cell(row=k, column=1).value = pid
        sheet.cell(row=k, column=2).value = lobe
        sheet.cell(row=k, column=3).value = round(dist_to_ventricle,2)

        k += 1

    book.save(folder + "brain_lobes_Hammers_mith_n30r95.xlsx")

    print(res_lobes_brain, len(res_lobes_brain))

if __name__ == "__main__":
    folder = "RES_survival_time_" + "{:%d%m%Y_%H%M}".format(datetime.datetime.now()) + "/"
    process_labels(folder)
    process(folder)
