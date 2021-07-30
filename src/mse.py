########################################################
##
## input is two M x N matrices (--matrix1 and --matrix2) with each matrix (1 and 2) having M=examples, N=features
## calculate mean square error between vectors (rows) in matrix 1 and matrix 2
## 
## example useage:
## python2.7 /p/lscratchf/allen99/lbexp/mse.py --matrix1 test1  --matrix2 test2 
##
########################################################
import sys
import numpy as np
import argparse

def main () :
   parser = argparse.ArgumentParser(description='Mean square error')
   parser.add_argument('--matrix1',required=True, help= "feature file (X)")
   parser.add_argument('--matrix2',required=True, help= "feature file (X)")

   args=parser.parse_args()

   ## initialize key parameters and input variables
   feat_file1=args.matrix1
   feat_mat1 =np.loadtxt(feat_file1)
   if args.matrix2 != "" :
      feat_file2=args.matrix2
      feat_mat2 =np.loadtxt(feat_file2)
   else :
      feat_mat2 = feat_mat1

   row,col = feat_mat1.shape

   iarr = np.arange( row )

   sumv,cnt = 0,0
   for v1i in range(row) :
      for v2i in range(row) :
         val=(feat_mat1[v1i]-feat_mat2[v2i])
         val*=val
         tval = sum(val)
         print "really?",tval
         sumv += tval
         cnt += 1

   rme = sumv / cnt
   print "RME",rme,cnt

if __name__ == '__main__' :
   main()
