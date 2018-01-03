# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:41:50 2016

@author: dahoiv
"""

# import os
import collections
import pickle
import util
import sqlite3
import nibabel as nib


def format_dict(d):
    d = collections.OrderedDict(sorted(d.iteritems()))
    s = ['lobe                   Type1   Type2   Type3 \n']
    for k, v in d.items():
        v = str(v[0]) + "      " + str(v[1]) + "      " + str(v[2])
        tab = 25 - len(k)
        s.append('%s%s %s\n' % (k, ' '*tab,  v))
    return ''.join(s) + '\n\n'


def process(folder):
    """ Post process data """
    print(folder)
    util.setup(folder, 'MolekylareMarkorer')
    conn = sqlite3.connect(util.DB_PATH, timeout=120)
    conn.text_factory = str
    cursor = conn.execute('''SELECT pid from MolekylareMarkorer ORDER BY pid''')
    image_ids = []
    tag_data_1 = []
    tag_data_2 = []
    tag_data_3 = []

    img = nib.load("/home/dahoiv/disk/data/MolekylareMarkorer/lobes_brain.nii")
    lobes_brain = img.get_data()
    label_defs = util.get_label_defs()
    res_right_left_brain = {}
    res_lobes_brain = {}
    patients = '\nPID  MM\n----------------\n'

    for pid in cursor:
        pid = pid[0]
        _id = conn.execute('''SELECT id from Images where pid = ?''', (pid, )).fetchone()
        if not _id:
            print("---No data for ", pid)
            continue
        _id = _id[0]

        _mm = conn.execute("SELECT Subgroup from MolekylareMarkorer where pid = ?",
                           (pid, )).fetchone()[0]
        if _mm is None:
            print("No mm data for ", pid)
            patients += str(pid) + ': ?\n'
            continue

        _desc = conn.execute("SELECT comments from MolekylareMarkorer where pid = ?",
                             (pid, )).fetchone()[0]
        if _desc is None:
            _desc = ""

        _filepath = conn.execute("SELECT filepath_reg from Labels where image_id = ?",
                                 (_id, )).fetchone()[0]
        if _filepath is None:
            print("No filepath for ", pid)
            continue

        com, com_idx = util.get_center_of_mass(util.DATA_FOLDER + _filepath)
        val = {}
        val['Name'] = str(pid) + "_" + str(_mm)
        val['PositionGlobal'] = str(com[0]) + "," + str(com[1]) + "," + str(com[2])
        val['desc'] = str(_desc)

        lobe = label_defs[lobes_brain[com_idx[0], com_idx[1], com_idx[2]]]
        right_left = 'left' if com_idx[0] < 99 else 'right'
        res_lobes_brain[lobe] = res_lobes_brain.get(lobe, [0, 0, 0])
        res_right_left_brain[right_left] = res_right_left_brain.get(right_left, [0, 0, 0])
        print(right_left, lobe)
        if _mm == 1:
            res_lobes_brain[lobe][0] += 1
            res_right_left_brain[right_left][0] += 1
        elif _mm == 2:
            res_lobes_brain[lobe][1] += 1
            res_right_left_brain[right_left][1] += 1
        elif _mm == 3:
            res_lobes_brain[lobe][2] += 1
            res_right_left_brain[right_left][2] += 1

        image_ids.extend([_id])
        print(pid, _mm)
        patients += str(pid) + ': ' + str(_mm) + '\n'
        if _mm == 1:
            tag_data_1.append(val)
        elif _mm == 2:
            tag_data_2.append(val)
        elif _mm == 3:
            tag_data_3.append(val)

    print(format_dict(res_lobes_brain))
    lobes_brain_file = open(folder + "lobes_brain.txt", 'w')
    lobes_brain_file.write(format_dict(res_lobes_brain))
    lobes_brain_file.close()
    lobes_brain_file = open(folder + "lobes_brain.txt", 'a')
    lobes_brain_file.write(format_dict(res_right_left_brain))
    lobes_brain_file.write(patients)
    lobes_brain_file.close()

    print(len(image_ids))
    return
    tag_data = {"tag_data_1": tag_data_1, "tag_data_2": tag_data_2, "tag_data_3": tag_data_3}
    pickle.dump(tag_data, open("tag_data.pickle", "wb"))

    cursor.close()
    conn.close()
    util.write_fcsv("mm_1", folder, tag_data_1, '1 0 0', 13)
    util.write_fcsv("mm_2", folder, tag_data_2, '0 1 0', 5)
    util.write_fcsv("mm_3", folder, tag_data_3, '0 0 1', 6)
    result = util.post_calculations(image_ids)
    util.avg_calculation(result['all'], 'all', None, True, folder, save_sum=True)
    util.avg_calculation(result['img'], 'img', None, True, folder)


if __name__ == "__main__":
    folder = "RES_MM/"
    process(folder)
