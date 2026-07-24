# Single-file optimized S4D galaxy classifier.
#   Host (x86, quick check):        make
#   Host per-layer counts:          enable perf (see README), then run galaxy_app
#   RISC-V real instret counts:     make bench CC=riscv32-unknown-elf-gcc CFLAGS="-O2"
#                                    (default arch already has the vector ext; do NOT pass -march=rv32gcv)
CC     ?= gcc
CFLAGS ?= -O2 -Wall

all: galaxy_app

galaxy_app: galaxy_s4d.c profile.h
	$(CC) $(CFLAGS) -o galaxy_app galaxy_s4d.c

# Baked build: weights + sample-0 image compiled in (bench_data.h), so it runs under
# qemu-riscv32 -- whose newlib libc cannot fopen files -- and reports real instret counts.
bench: galaxy_s4d.c bench_data.h profile.h
	$(CC) $(CFLAGS) -DBAKED -o galaxy_bench galaxy_s4d.c

clean:
	rm -f galaxy_app galaxy_bench *.o

.PHONY: all bench clean
