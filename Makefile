all : a.out result.txt
a.out : main3V-FK.cu FK-3V-master.cu globalVariables.cuh hostPrototypes.h typedef3V-FK.h \
	devicePrototypes.cuh helper_functions.cu openGLPrototypes.h \
	volRenderCuda.cu printFunctions.cu SOIL.h
	nvcc main3V-FK.cu FK-3V-master.cu helper_functions.cu volRenderCuda.cu  \
	printFunctions.cu -arch=sm_61 -rdc=true -lglut -lGL -lGLEW \
	-lSOIL -std=c++11
result.txt : a.out
	rm -rf result.txt
	./a.out
