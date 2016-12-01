# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:41:50 2016

@author: dahoiv
"""

# import os
import util
import sqlite3

def process(folder):
    print(folder)
    util.setup(folder, 'MolekylareMarkorer')
    conn = sqlite3.connect(util.DB_PATH, timeout=120)
    conn.text_factory = str
    cursor = conn.execute('''SELECT pid from MolekylareMarkorer''')
    image_ids = []
    tag_data = []
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

        _filepath = conn.execute("SELECT filepath_reg from Labels where image_id = ?",
                                (_id, )).fetchone()[0]
        if _filepath is None:
            print("No filepath for ", _id)
            continue

        com = util.get_center_of_mass(util.DATA_FOLDER + _filepath)
        val = {}
        val['Name'] = str(pid) + "_" + str(_mm)
        val['PositionGlobal'] = str(com[0]) + "," + str(com[1]) + "," + str(com[2]) 


        image_ids.extend([_id])
        tag_data.append(val)
        util.write_fcsv(folder + "test.fcsv", tag_data)
        return
        
    cursor.close()
    conn.close()
    
    result = util.post_calculations(image_ids)
    util.avg_calculation(result['all'], 'all', None, True, folder, save_sum=True)
    util.avg_calculation(result['img'], 'img', None, True, folder)


if __name__ == "__main__":
    folder = "RES_MM/"
    process(folder)
