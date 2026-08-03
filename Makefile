# Single-file optimized S4D galaxy classifier.
#   Host (x86, quick check):        make
#   Host per-layer counts:          enable perf (see README), then run build/main
#   RISC-V real instret counts:     make bench CC=riscv32-unknown-elf-gcc CFLAGS="-O2"
#                                    (default arch already has the vector ext; do NOT pass -march=rv32gcv)
CC       ?= gcc
CFLAGS   ?= -O2 -Wall
BUILDDIR := build

all: $(BUILDDIR)/main

$(BUILDDIR)/main: main.c profile.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ main.c

# Baked build: weights (weights.h) + sample-0 image (image.h) compiled in, so it runs under
# qemu-riscv32 -- whose newlib libc cannot fopen files -- and reports real instret counts.
bench: $(BUILDDIR)/bench

$(BUILDDIR)/bench: main.c weights.h image.h profile.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -DBAKED -o $@ main.c

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)

.PHONY: all bench clean
