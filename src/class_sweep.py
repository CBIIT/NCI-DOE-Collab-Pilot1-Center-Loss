###
###
###  Python script to launch a simple sweep of hyper paramters
###
###
import os,sys
from subprocess import call

#nodes=int(sys.argv[1])   ## number of compute nodes to use
partitions=int(sys.argv[1])  ## expected number of cross validation partitions
## name of the train/testing cross validation files (format is: "filebn".train.fea.X or "filebn".test.fea.X)
## where 
filebn=sys.argv[2] 
ddir=sys.argv[3]

## original
## ddir="/p/lscratchf/allen99/anlftp/public/datasets/GDC/data_frames/BySite"

aecmd="/p/lscratchf/allen99/lbexp/classify1a.py"

##source code command line references
##LearnRateMethod = Input("--learning-rate-method", "1 - Adagrad, 2 - RMSprop, 3 - Adam", LearnRateMethod);
##ActivationType = static_cast<activation_type>(Input("--activation-type", "1 - Sigmoid, 2 - Tanh, 3 - reLU, 4 - id", static_cast<int>(ActivationType)));

## -f -> data location
## -e -> epoch
## -b -> mini-match
## -a -> activiation type 
## -r -> learning rate
## -j -> learning rate decay
## -k -> fraction of training data to use for training
## -g -> dropout rate
## -q -> learning rate method
## -n -> network topology : specify number of nodes in each hidden layer
## original parameters

num_feat=[0,100,1000,1000]
labels=range(27)

for nf in num_feat : 
   if nf > 0 :
      fstr="--rand_feat "+str(nf) +" "
   else :
      fstr=""
   for label in labels : 
      params=fstr+"--label "+str(label)
      param_lst.append(params)

for hpi in range(len(param_lst)) :
   for parti in range(partitions) :
      #tr_file=ddir+filebn+".train.fea."+str(parti)
      trainy=ddir+filebn+".test.lab."+str(parti)
      trainy=ddir+filebn+".test.fea."+str(parti)
      out_name="ae_log."+str(os.getpid())+".out."+str(parti) + ".hp."+str(hpi)
      #run_cmd="sbatch -N"+str(nodes) + " -t 1440 --clear-ssd --msr-safe --output="+out_name+" "+aecmd+" -x "+tr_file+" -y " +ts_file+" "+param_lst[hpi]

      run_cmd="aecmd+" --trainy "+tr_file+" -y " +ts_file+" "+param_lst[hpi]
      print hpi
      print run_cmd
      #call(run_cmd,shell=True)
