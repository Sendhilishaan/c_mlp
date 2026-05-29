CC = gcc
CFLAGS = -Wall -Wextra -g -Isrc

build: src/math/matrix.c src/nn/nn.c src/main.c src/math/functions.c
	$(CC) $(CFLAGS) src/nn/nn.c src/math/matrix.c src/math/functions.c src/main.c -o camel -Iinclude

clean:
	del camel.exe
