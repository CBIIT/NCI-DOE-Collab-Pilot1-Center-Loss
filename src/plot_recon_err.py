###
### Simple matplotlib routine to plot reconstruction error (y-axis) with standard deviation over epochs (x-axis) 
### 
###
### example useage for 3 different autoencoder models:
### python3 ../plot_recon_err.py "e_log.20468.out.hp.0,400x300x100 e_log.20468.out.hp.1,500x100 e_log.20468.out.hp.2,1000x500" ex1.png
import sys
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='Plot mean square error')
parser.add_argument('--files',required=False, help= "MSE files")
parser.add_argument('--file_list',required=False, help= "list of MSE files")
parser.add_argument('--output',required=True, help= "Output PNG file")
parser.add_argument('--nostdev',dest='nostdev', action='store_true',help= "plot stdev")
parser.add_argument('--ylog',dest='ylog', action='store_true',help= "plot stdev")
parser.set_defaults(nostdev=False)
parser.set_defaults(ylog=False)
args=parser.parse_args()

output=args.output
flst=""
if args.files :
   flst=args.files
elif args.file_list :
   fh=open(args.file_list)
   cnt = 0
   for line in fh :
      if cnt > 0 :
         flst += " "
      line=line.rstrip()
      flst += line
      cnt += 1
assert flst != ""
files=[]
print(flst)
if flst.find(" ") != -1 :
      files=flst.split(" ")
else :
      files.append(flst)

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
lstyle='dotted'
cval=["red","blue","green","purple","orange","black"]
cnt=0
for file_all in files :
   file,desc=file_all.split(',')
   fh=open(file)
   xval,yval,std=[],[],[]
   for line in fh :
      line=line.rstrip()
      vals=line.split(" ")
      xval.append(int(vals[0]))
      yval.append(float(vals[1]))
      if args.nostdev :
         std.append(0)
      else :
         std.append(float(vals[2]))

   if cnt >= len(cval) :
      cnt = 0
   plt.errorbar(xval,yval,yerr=std,ls=lstyle,color=cval[cnt],label=desc)
   if args.ylog :
      ax=plt.subplot()
      ax.set_yscale("log")
   cnt+=1

plt.xlabel('Epoch')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.savefig(output,dpi=100)
plt.show()
