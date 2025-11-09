# Simple Makefile for spmv project
CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -march=native -fopenmp
LDLIBS ?=
OUT ?= executable
SRCS := main.c generator.c m_to_csr.c
OBJS := $(SRCS:.c=.o)

# # detect OpenMP usage and add flags (best-effort)
# ifeq (, $(filter %#include <omp.h>%,$(shell sed -n '1,200p' $(SRCS) 2>/dev/null)))
# # no-op
# else
# CFLAGS += -fopenmp
# LDLIBS += -fopenmp
# endif

.PHONY: all clean run

all: $(OUT)

$(OUT): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# generic rule to compile .c -> .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(OUT)

run: all
	./$(OUT)
