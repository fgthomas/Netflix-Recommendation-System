"""
@file getvectors.py
@author Forest Thomas
@brief A class to fetch vectors for use with a neural net
"""

import sqlite3

class Fetcher:
    """
    Class to keep track of where in the database we are
    """
    def __init__(self):
        self.next = -1
        self.conn = sqlite3.connect("users.db")
        self.c = self.conn.cursor()
        self.c2 = self.conn.cursor()
       

    def getNext(self):
        self.c.execute("SELECT * FROM users0 WHERE uid > {} ORDER BY uid".format(self.next))
        vec = self.c.fetchone()
        if vec == None:
            self.next = -1
            return getNext()
        for i in range(1, 18):
            self.c2.execute("SELECT * FROM users{0} WHERE uid = {1}".format(i, vec[0]))
            vec = vec + self.c2.fetchone()[1:]
        self.next = vec[0]
        return vec

    def finish(self):
        self.conn.close()

f = Fetcher()

def getNext():
    return f.getNext()

def getFormattedVector():
    a = getNext()[1:]
    v = []
    
    for i in range(0, len(a)):
        if a[i] == None:
            v.append(None)
        else:
            vec = [0, 0, 0, 0, 0]
            vec[a[i] - 1] = 1
            v.append(vec)
    return v
    
def getFrom(n):
    f.next = n-1
    a = getNext()[1:]
    v = []
    for i in range(0, len(a)):
        if a[i] == None:
            v.append(None)
        else:
            vec = [0, 0, 0, 0, 0]
            vec[a[i] - 1] = 1
            v.append(vec)
    return v

def finish():
    f.close()
