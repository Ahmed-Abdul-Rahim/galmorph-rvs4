CC     ?= gcc
CFLAGS ?= -O2 -Wall
BUILD  := build
all: $(BUILD)/galaxy
$(BUILD)/galaxy: galaxy_cnn.c profile.h | $(BUILD)
	$(CC) $(CFLAGS) -w -o $@ galaxy_cnn.c
bench: $(BUILD)/bench
$(BUILD)/bench: galaxy_cnn.c profile.h | $(BUILD)
	$(CC) $(CFLAGS) -w -o $@ galaxy_cnn.c
bench-scalar: $(BUILD)/bench-scalar
$(BUILD)/bench-scalar: galaxy_cnn.c profile.h | $(BUILD)
	$(CC) $(CFLAGS) -w -DFORCE_SCALAR -o $@ galaxy_cnn.c
$(BUILD):
	mkdir -p $(BUILD)
clean:
	rm -rf $(BUILD)
.PHONY: all bench bench-scalar clean
