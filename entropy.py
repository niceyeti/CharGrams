import math
import sys

#given some ngram file, returns the total entropy of the set.
# REMEMBER readlines() only reads a few thousand (4-8k) lines; larger files require c++ or other modifications


def main():
  
  if len(sys.argv) <= 1:
    print "no args given. usage: entropy.py fname"
    return -1;  

  print "calculating entropy of ngrams in file: ", sys.argv[1]

  ifile = open(sys.argv[1],"r")

  lines = ifile.readlines()

  numItems = 0.0
  probX = 0.0
  totalEntropy = 0.0

  for gstr in lines:
    numItems += float(gstr.split("|")[1])
  print "numItems=", numItems

  totalEntropy = 0.0
  for gstr in lines:
    probX = float(gstr.split("|")[1]) / numItems
    totalEntropy += (probX * math.log(probX,2))
  totalEntropy = totalEntropy * -1


  print "Total entropy: ",totalEntropy

  ifile.close()

  return totalEntropy



if __name__ == '__main__':
  main()

