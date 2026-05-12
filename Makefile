CC = gcc
CFLAGS = -Wall -Wextra -g

build: src/math/matrix.c src/main.c
	$(CC) $(CFLAGS) src/math/matrix.c src/main.c -o cgrad

clean:
	del cgrad.exe
