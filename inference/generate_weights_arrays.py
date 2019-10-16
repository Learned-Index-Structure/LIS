import sys

headerfilein = "weights.h.ph"
headerfileout = "weights.h"
weightsfile = sys.argv[1]

headerfout = open(headerfileout, "w+")
headerfin = open(headerfilein, "r")
weightsf = open(weightsfile, "r")

headerfcontent = headerfin.read()

weightstring = weightsf.readlines()
w1 = weightstring[0].split()

weightsarrstring = "float w1[] = {" + ", ".join(w1) + " };"

w2 = ""
for i in range(32):
    w2 += "{" + ", ".join(weightstring[i+2].split()) + " }"
    if i != 31:
        w2 += ",\n"

w2 = "float w2[32][32] = {" + w2 + "};"
weightsarrstring += "\n\n" + w2

w3 = weightstring[35].split()
w3 = "float w3[] = {" + ", ".join(w3) + " };"
weightsarrstring += "\n\n" + w3

headerfcontent = headerfcontent.replace("$$", weightsarrstring)
headerfout.write(headerfcontent)