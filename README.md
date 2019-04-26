# Tune-DE-DDQN

# training_set is generated using following commands:
x = np.random.choice(2160, 1080,replace = False)

with open('training_set','w+') as f:
    
    for i in x:
        
        f.write(str(i)+'\n')

# CEC2014 functions (http://web.mysites.ntu.edu.sg/epnsugan/PublicSite/Shared%20Documents/CEC-2014/Definitions%20of%20%20CEC2014%20benchmark%20suite%20Part%20A.pdf):
Unimodal functions:

(1) Rotated High Conditioned Elliptic Function 

(2) Rotated Bent Cigar Function 

(3) Rotated Discus Function 

Simple multimodal functions:

(4) Shifted and Rotated Rosenbrock’s Function

(5) Shifted and Rotated Ackley’s Function

(6) Shifted and Rotated Weierstrass Function

(7) Shifted and Rotated Griewank’s Function 

(8) Shifted Rastrigin’s Function 

(9) Shifted and Rotated Rastrigin’s Function

(10) Shifted Schwefel’s Function 

(11) Shifted and Rotated Schwefel’s Function

(12) Shifted and Rotated Katsuura Function

(13) Shifted and Rotated HappyCat Function

(14) Shifted and Rotated HGBat Function 

(15) Shifted and Rotated Expanded Griewank’s plus Rosenbrock’s Function

(16) Shifted and Rotated Expanded Scaffer’s F6 Function 

Hybrid function1:

(17) Hybrid Function1 

(18) Hybrid Function2 

(19) Hybrid Function3 

(20) Hybrid Function4 

(21) Hybrid Function5 

(22) Hybrid Function6  

Composition functions:

(23) Composition Function1 

(24) Composition Function2 

(25) Composition Function3 

(26) Composition Function4 

(27) Composition Function5 

(28) Composition Function6 

(29) Composition Function7 

(30) Composition Function8 


# BBOB functions (http://coco.gforge.inria.fr/downloads/download16.00/bbobdocfunctions.pdf):
Separable functions:

(1) Sphere function

(2) Ellipsoidal function

(3) Rastrigin function

(4) Buche-Rastrigin Function

(5) Linear slope

Function with low or moderate conditiong:

(6) Attractive sector function

(7) Step ellipsoidal function

(8) Rosenbrock function, original

(9) Rosenbrock function, rotated

Functions with highconditioning and unimodal:

(10) Ellipsoidal function

(11) Discus function

(12) Bent cigar function


(13) Sharp ridge function

(14) Different powers function

Multi-modal functions with adequate global structure:

(15) Rastrigin function

(16) Weierstrass Function

(17) Schaffers F7 Function

(18) Schaffers F7 Function, moderately ill-conditioned

(19) Composite Griewank-Rosenbrock Function F8F2

Multi-modal functions with weak global structure

(20) Schewefel function

(21) Gallagher’s Gaussian 101-me Peaks Function

(22) Gallagher’s Gaussian 21-hi Peaks Function

(23) Katsuura function

(24) Lunacek bi-Rastrigin Function





























