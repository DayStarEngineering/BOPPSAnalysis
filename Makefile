# Compile Flags:
CPP = /usr/bin/g++-4.7
CPPFLAGS = -O3 setConstrainedParam -m64 -pipe -c -Wall -Wextra
LIBFLAGS = -O3 setConstrainedParam -m64 -pipe -c -Wall -Wextra -fPIC
LDFLAGS = setConstrainedParam -m64 -pipe -Wall -Wextra -lpthread -pthread -lrt

all: dep  

dep:
	make -C predictiveFilter
	make -C Centroid
	
clean:
	make -C Centroid clean
	make -C predictiveFilter clean