#!/bin/bash
clear
c=1
d=1
e=1
popu=10
gera=2

while [ $e -le 5 ] 
do
	if [ ! -d "pop$popu" ]; then
		mkdir "pop$popu";
	fi

		while [ $d -le 3 ]
		do

			if [ ! -d "pop$popu/ger$gera" ]; then
				mkdir "pop$popu/ger$gera";
			fi

				while [ $c -le 5 ]
				do
					if [ ! -d "pop$popu/ger$gera/$c" ]; then
						mkdir "pop$popu/ger$gera/$c";
						xargs -P 4 ./svm-train $popu $gera
						
						mv "results/mean.txt" "pop$popu/ger$gera/$c/mean$popu\_$gera.txt"
						mv "results/stdev.txt" "pop$popu/ger$gera/$c/stdev$popu\_$gera.txt"
						mv "results/best.txt" "pop$popu/ger$gera/$c/best$popu\_$gera.txt"
						mv "results/best_complete.txt" "pop$popu/ger$gera/$c/best\_complete$popu\_$gera.txt"
																		
					fi

					(( c++ ))
				done

		((gera +=  10));
		(( d++ ))
		c=1

	   done
	
	((popu +=  + 20));
	(( e++ ))
	gera=10
	
done


