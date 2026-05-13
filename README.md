=================================================== <br>
# MFSSG
This is the Python implementation of the MFSSG algorithm. Note that this implementation can employ GPU to speed up the code.

<B> 1. Enviroment and dependent packages </B> <br>
MFSSG was developed in conda enviroment (python 3.11).<br>
The dependent packages includes: </br>
Numpy 1.26.2;  <br>
torch 2.1.1; <br>
scipy 1.16.3; <br>
sklearn 1.8.0; <br>
mgm 0.5.8 <br>

<B> 2.Datasets </B> </br> 
This folder includes ESRD to train MFSSG. The dataset is publicly available from the directory "data/ESRD.mat".

<B> 3. Usage Instructions </B> </br> 
MFSSG includes the main functions below: 
run.py </br> 
This is the implementation function of MFSSG, which provides an illustrative example of MFSSG on ESRD dataset. 

model.py </br> 
This is the main code, where the MFSSG model are designed, built and trained, and finally the results are returned. 

tools.py </br> 
This file contains some external functions, including data loading, preprocessing, factors initition, SNF, distance computation and so on.

evl.py </br> 
This file contains some evaluation metrics, sucn as AC, NMI, ARI, Purity from sklearn. </br> 
