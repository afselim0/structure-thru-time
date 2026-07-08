#!/bin/bash

for i in ../results/simulations/outputs/*; 
do 
    echo $i; 
    for j in $i/sim/plots/*.{html,png,pdf,sh}; 
        do 
        echo $j; 
        rsync -a -- "$j" "../results/simulations/fig_scrape/${i##*/}_${j##*/}"
    done;
   

done;
