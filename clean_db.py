# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:32:12 2016

@author: dahoiv
"""

import os
import sqlite3

import util

if __name__ == "__main__":
    os.nice(19)

    util.setup("temp/", "")

    all_filepaths = ['brainSegmentation.db']

    conn = sqlite3.connect(util.DB_PATH)
    conn.text_factory = str
    cursor = conn.execute('''SELECT filepath, transform, filepath_reg from Images''')
    for (filepath, transform, filepath_reg) in cursor:
        all_filepaths.extend(util.ensure_list(filepath))
        all_filepaths.extend(util.ensure_list(filepath_reg))
        if transform is None:
            continue
        for _transform in transform.split(","):
            all_filepaths.append(_transform.strip())

    cursor = conn.execute('''SELECT filepath, filepath_reg from Labels''')
    for (filepath, filepath_reg) in cursor:
        all_filepaths.extend(util.ensure_list(filepath))
        all_filepaths.extend(util.ensure_list(filepath_reg))

    for root, dirs, files in os.walk(util.DATA_FOLDER):
        for filepath in files:
            filepath = os.path.join(root, filepath).replace(util.DATA_FOLDER, "")
            if filepath not in all_filepaths:
                print("Delete " + filepath)
            else:
                all_filepaths.remove(filepath)

    print("Delete ", all_filepaths)