CC = gcc
CFLAGS = -Wall -Wextra -g

build: src/matrix.c src/main.c
	$(CC) $(CFLAGS) src/matrix.c src/main.c -o cgrad

clean:
	del cgrad.exe
