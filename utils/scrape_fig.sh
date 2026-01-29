#!/bin/bash

for i in outputs/*; 
do 
    echo $i; 
    for j in $i/plots/*.{html,png,pdf}; 
        do 
        echo $j; 
        rsync -a -- "$j" "fig_scrape/${i##*/}_${j##*/}"
    done;
done;
