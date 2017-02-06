# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:41:50 2016

@author: dahoiv
"""

# import os
import pickle
import util
import sqlite3


def process(folder):
    print(folder)
    util.setup(folder, 'MolekylareMarkorer')
    conn = sqlite3.connect(util.DB_PATH, timeout=120)
    conn.text_factory = str
    cursor = conn.execute('''SELECT pid from MolekylareMarkorer''')
    image_ids = []
    tag_data_1 = []
    tag_data_2 = []
    tag_data_3 = []
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
            print("No mm data for ", _id)
            continue

        _desc = conn.execute("SELECT comments from MolekylareMarkorer where pid = ?",
                             (pid, )).fetchone()[0]
        if _desc is None:
            _desc = ""

        _filepath = conn.execute("SELECT filepath_reg from Labels where image_id = ?",
                                 (_id, )).fetchone()[0]
        if _filepath is None:
            print("No filepath for ", _id)
            continue

        com = util.get_center_of_mass(util.DATA_FOLDER + _filepath)
        val = {}
        val['Name'] = str(pid) + "_" + str(_mm)
        val['PositionGlobal'] = str(com[0]) + "," + str(com[1]) + "," + str(com[2])
        val['desc'] = str(_desc)

        image_ids.extend([_id])
        print(_mm)
        if _mm == 1:
            tag_data_1.append(val)
        elif _mm == 2:
            tag_data_2.append(val)
        elif _mm == 3:
            tag_data_3.append(val)

    tag_data = {"tag_data_1": tag_data_1, "tag_data_2": tag_data_2, "tag_data_3": tag_data_3}
    pickle.dump(tag_data, open("tag_data.pickle", "wb"))

    cursor.close()
    conn.close()
    util.write_fcsv(folder + "mm_1.fcsv", tag_data_1, '1,0,0')
    util.write_fcsv(folder + "mm_2.fcsv", tag_data_2, '0,1,0')
    util.write_fcsv(folder + "mm_3.fcsv", tag_data_3, '0,0,1')
    result = util.post_calculations(image_ids)
    util.avg_calculation(result['all'], 'all', None, True, folder, save_sum=True)
    util.avg_calculation(result['img'], 'img', None, True, folder)


if __name__ == "__main__":
    folder = "RES_MM/"
    process(folder)
