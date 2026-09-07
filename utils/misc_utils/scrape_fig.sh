#!/bin/bash

for i in ../../results/outputs/simulations/*; 
do 
    echo $i; 
    for j in $i/sim/plots/*.{html,png,pdf,sh}; 
        do 
        echo $j; 
        rsync -a -- "$j" "../../results/figures/sim_figs_png/${i##*/}_${j##*/}"
    done;
   

done;
