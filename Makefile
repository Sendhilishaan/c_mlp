CC = gcc
CFLAGS = -Wall -Wextra -g -Isrc

build: src/math/matrix.c src/nn/nn.c src/main.c src/math/functions.c src/api.c
	$(CC) $(CFLAGS) src/nn/nn.c src/math/matrix.c src/math/functions.c src/main.c src/api.c -o camel -Iinclude

dll: src/math/matrix.c src/nn/nn.c src/math/functions.c src/api.c
	$(CC) $(CFLAGS) -shared -o camel.dll src/nn/nn.c src/math/matrix.c src/math/functions.c src/api.c -Iinclude

clean:
	del camel.exe camel.dll
