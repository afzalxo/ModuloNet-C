all:
	gcc -o modulonet_run main.c modulonet.c layer_ops.c utils.c matmul.c modulonet.h layer_ops.h globals.h utils.h matmul.h

clean:
	rm modulonet_run
