import numpy as np

def coord_flat2tuple(coord):
    if coord==81: return None
    return coord//9, coord%9

def coord_tuple2flat(coord):
    if coord==None: return 81
    return coord[0]*9+coord[1]

colormap = {
        'white': 1,
        'black': -1,
        'empty': 0
}
