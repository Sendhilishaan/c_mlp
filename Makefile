CC = gcc
CFLAGS = -Wall -Wextra -g -Isrc

build: src/math/matrix.c src/nn/nn.c src/main.c
	$(CC) $(CFLAGS) src/nn/nn.c src/math/matrix.c src/main.c -o cgrad

clean:
	del cgrad.exe
