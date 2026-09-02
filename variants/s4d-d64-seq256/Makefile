CC       ?= gcc
CFLAGS   ?= -O2 -Wall
BUILDDIR := build
all: $(BUILDDIR)/main
$(BUILDDIR)/main: main.c profile.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ main.c
bench: $(BUILDDIR)/bench
$(BUILDDIR)/bench: main.c weights.h image.h profile.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -DBAKED -o $@ main.c
bench-scalar: $(BUILDDIR)/bench-scalar
$(BUILDDIR)/bench-scalar: main.c weights.h image.h profile.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -DBAKED -DFORCE_SCALAR -o $@ main.c
$(BUILDDIR):
	mkdir -p $(BUILDDIR)
clean:
	rm -rf $(BUILDDIR) weights.h weights.bin image.h image.bin
.PHONY: all bench bench-scalar clean
