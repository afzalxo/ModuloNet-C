CC=gcc
CFLAGS=-I.
DEPENDENCIES=utils.h globals.h layer_ops.h matmul.h modulonet.h

%.o: %.c $(DEPENDENCIES)
	$(CC) -c -o $@ $< $(CFLAGS)
modnet_run: main.o utils.o layer_ops.o matmul.o modulonet.o globals.h
	$(CC) -o modnet_run main.o utils.o layer_ops.o matmul.o modulonet.o globals.h
