# Single-file optimized S4D galaxy classifier.
#   Host (x86, quick check):   make
#   RISC-V + vector kernels:   make CC=riscv32-unknown-elf-gcc CFLAGS="-O2 -march=rv32gcv -mabi=ilp32f"
CC     ?= gcc
CFLAGS ?= -O2 -Wall

all: galaxy_app

galaxy_app: galaxy_s4d.c profile.h
	$(CC) $(CFLAGS) -o galaxy_app galaxy_s4d.c

clean:
	rm -f galaxy_app *.o

.PHONY: all clean
