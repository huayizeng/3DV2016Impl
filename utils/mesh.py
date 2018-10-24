import re
import os
import numpy as np
from IPython import embed

def read_obj(filename):
    f = open(filename, "r") 
    if not f:
        print ("Cannot open a file: " + filename)
        return None
    vertices = []
    faces = []

    v_pattern = re.compile(r"v\s+([+-]?\d+[.]*\d*)\s+([+-]?\d+[.]*\d*)\s+([+-]?\d+[.]*\d*)\s+([+-]?\d+[.]*\d*)")
    f_pattern = re.compile(r"f\s+(\d+[.]*\d*)\s+(\d+[.]*\d*)\s+(\d+[.]*\d*)")
    for line in f:
        if line[0] == "v" and line[1] == " ":

            m = line.split()
            vertices.append( [ float(m[1]), float(m[2]), float(m[3]) ] )

            # m = v_pattern.search(line)
            # if m:
            #     vertices.append( [ float(m.group(1)), float(m.group(2)), float(m.group(3)) ] )

        if line[0] == "f":
            m = f_pattern.search(line)
            if m:
                faces.append([int(m.group(1))-1, int(m.group(2))-1, int(m.group(3))-1])
    f.close()
    return (vertices, faces)

def write_obj(filename, vertices, faces):
    with open(filename, "w+") as fout:          
        for v in vertices:
            fout.write("v " + str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + "\n")
        for f in faces:
            fout.write("f " + str(f[0] + 1) + " " + str(f[1] + 1) + " " + str(f[2] + 1) + "\n")
    return True