CC     ?= gcc
CFLAGS ?= -O2 -Wall
BUILD  := build
all: $(BUILD)/cnn
$(BUILD)/cnn: cnn_opt.c config.h image_data.h profile.h | $(BUILD)
	$(CC) $(CFLAGS) -w -o $@ cnn_opt.c
bench: $(BUILD)/bench
$(BUILD)/bench: cnn_opt.c config.h image_data.h profile.h | $(BUILD)
	$(CC) $(CFLAGS) -w -o $@ cnn_opt.c
bench-scalar: $(BUILD)/bench-scalar
$(BUILD)/bench-scalar: cnn_opt.c config.h image_data.h profile.h | $(BUILD)
	$(CC) $(CFLAGS) -w -DFORCE_SCALAR -o $@ cnn_opt.c
$(BUILD):
	mkdir -p $(BUILD)
clean:
	rm -rf $(BUILD)
.PHONY: all bench bench-scalar clean
