# ds4-dump-logits — captures full per-token logits as the Phase 1 reference.
#
# Links against ds4's object files (built via `make -C ../ds4 rocm`) so we
# don't need to recompile ds4. The build expects ROCM_PATH to be set (via the
# deepstrix nix dev shell) and ds4 to already be built.

DS4_DIR ?= ../ds4
ROCM_ARCH ?= gfx1151

CC := gcc
CFLAGS := -O3 -ffast-math -Wall -Wextra -std=c99 -D_GNU_SOURCE -fno-finite-math-only -I$(DS4_DIR)
HIPCC := $(ROCM_PATH)/bin/hipcc
HIPLDFLAGS := --offload-arch=$(ROCM_ARCH) -L$(ROCM_PATH)/lib -lhipblas -lm -pthread

DS4_OBJS := $(DS4_DIR)/ds4.o $(DS4_DIR)/ds4_cuda.o

.PHONY: all clean

all: ds4-dump-logits

# Build ds4's .o files if missing, by delegating to its Makefile.
$(DS4_DIR)/ds4.o $(DS4_DIR)/ds4_cuda.o:
	$(MAKE) -C $(DS4_DIR) rocm ROCM_ARCH=$(ROCM_ARCH)

dump_logits.o: dump_logits.c $(DS4_DIR)/ds4.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Link via hipcc so the device-side runtime is properly bound.
ds4-dump-logits: dump_logits.o $(DS4_OBJS)
	$(HIPCC) $(HIPLDFLAGS) -o $@ $^

clean:
	rm -f dump_logits.o ds4-dump-logits
