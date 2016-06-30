# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:26:32 2016

@author: dahoiv
"""

import sqlite3


class img_data(object):
    def __init__(self, image_id, data_path):
        self.image_id = image_id
        self.data_path = data_path
        
        self.bet_filepath = None
        self.init_transform = None
        
        self._img_filepath = None
        self._label_filepath = None
        
    @property
    def img_filepath(self):
        if not self._img_filepath is None:
            return self._img_filepath
        conn = sqlite3.connect(self.db_path)
        conn.text_factory = str
        cursor = conn.execute('''SELECT filepath from Images where id = ? ''', (self.image_id,))
        self._img_filepath =  self.data_path + cursor.fetchone()[0]
        cursor.close()
        conn.close()

        return self._img_filepath
        
    @property
    def label_filepath(self):
        if not self._label_filepath is None:
            return self._label_filepath
        
        conn = sqlite3.connect(self.db_path)
        conn.text_factory = str
        cursor = conn.execute('''SELECT filepath from Labels where image_id = ? ''',
                              (self.image_id,))
        self._label_filepath =  self.data_path + cursor.fetchone()[0]
        cursor.close()
        conn.close()    
        
        return self._label_filepath
        
    @property        
    def db_path(self):
        return self.data_path + "brainSegmentation.db"
    