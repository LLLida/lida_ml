# Builds the WEED text editor.

CC ?= gcc

CFLAGS ?= -O0 -g3 -Wall -Wextra -Wpedantic -Wno-unused-function
LDFLAGS ?=

# uncomment if using ASAN
# CFLAGS += -fsanitize=address -fno-omit-frame-pointer
# LDFLAGS += -fsanitize=address -fno-omit-frame-pointer -lrt

.PHONY: all clean

all: main

main: lida_ml.h lida_ml.c main.c
	$(CC) $(CFLAGS) main.c lida_ml.c $(LDFLAGS) -o $@
