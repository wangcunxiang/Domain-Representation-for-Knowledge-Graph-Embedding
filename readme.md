# DRE: Domain Representation in Knowledge Graph Embedding 

Source code for DRE. 

Data sets are WN18 and FB15K.

## Usage

### Compile the program

We use  `g++` to compile in the program. 

```
g++ X.cpp -o X -O3 -pthread -march=native
train: X = ellipsoid
test: X = test_ellipsoid/test_R_ellipsoid/test_ST_ellipsoid
```

### Run the program

####Train

The program combines training of TransE/TransR/STransE and DRE(TransE)/DRE(TransR)/DRE(STransE) in ellipsoid.cpp file. You can choose the baseline model from TransE/TransR/STransE, and the program will train the baseline model and the ellipsoids automatically.

To run the program, we perform:

```
$./ellipsoid -isTransX <0/1/2> -withEllipsoid <0/1> -initwithTransE <0/1> -input <dir> -output <dir> -size <int> -sizeR <int> -margin_E <float> -margin_R <float> -margin_S <float> -epochs_E <int> -epochs_R <int> -epochs_S <int> -alpha <float> -alpha <float> -rate <float> -rate <float> -note <string>
```

**parameters:**

`-isTransX <0/1/2>`:  1: train TransR and use the results to train ellipsoids. 2: train STransE and use the results to train ellipsoids. 0: only train TransE and use the results to train ellipsoid.

`-withEllipsoid <0/1>`:  0: not train ellipsoids. 1: train ellipsoids. 

`-initwithTransE <0/1>`:  0: not init with TransE. 1: init with TransE ( when training TransR/STransE)

`-input <dir>`:  path to the dataset directory.

`-output <dir>`: path to the output directory.

`-size <int>`: the number of entity vector dimensions. 

`-sizeR <int>`: the number of relation vector dimensions. 

`-margin_E <float>`: the margin hyper-parameter for TransE.

`-margin_R <float>`: the margin hyper-parameter for TransR.

`-margin_S <float>`: the margin hyper-parameter for STransE.

`-epochs_E <int>`: the epochs for TransE.

`-epochs_R <int>`: the epochs for TransR.

`-epochs_S <int>`: the epochs for STransE.

`-alpha <float>`: the SGD learning rate for TransE, TransR and STransE.

`-rate <float>`: the SGD learning rate for ellipsoid.

`-thread <int>`: the number of worker threads.

`-note <string>`: information you want to add in the filename.

#### test

For evaluating link/entity prediction, the program provides  meanrank, Hits@1, Hits@5 and Hits@10 in two setting  protocols "Raw" and "Filtered". The testing codes are divided to 3 files, test_ellipsoid.cpp, test_R_ellipsoid.cpp and test_ST_ellipsoid.cpp respectively. Each for testing DRE(TransE), DRE(TransR) and DRE(STransE).

```
$ ./test_ellipsoid -domain <0/1> -size <int> -sizeR <int> -input <dir> -init <dir> -load <dir> -thread <int> -note <string>

$ ./test_R_ellipsoid -domain <0/1> -size <int> -sizeR <int> -input <dir> -init <dir> -load <dir> -thread <int> -note <string>

$ ./test_ST_ellipsoid -domain <0/1> -size <int> -sizeR <int> -input <dir> -init <dir> -load <dir> -thread <int> -note <string>
```
**parameters:**

`-domain <0/1>`:  0: test without domain; 1: test with domain.

`-size <int>`: the number of entity vector dimensions. 

`-sizeR <int>`: the number of relation vector dimensions. 

`-input <dir>`:  path to the dataset directory.

`-init <dir>`:  path to the baseline model vectors directory.

`-load <dir>`:  path to the ellipsoid vectors directory.

`-thread <int>`: the number of worker threads.

`-note <string>`: information you want to add in the filename. (\_TransE/\_TransR/\_STransE are needed for regular output by ellipsoid.cpp)

**Test Results**

After 10/3/1: , the first four lines are for overall results; 5-8 lines are for 1-to-1 relations; 9-12 lines are for 1-to-N results; 13-16 lines are for N-to-1 results ; 17-20 lines are for N-to-N results; 21-24 lines are for type constraint results  