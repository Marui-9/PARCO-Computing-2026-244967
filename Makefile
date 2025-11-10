# Simple Makefile for spmv project
CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -march=native -fopenmp
LDLIBS ?= -lm
OUT ?= executable
TEST_OUT ?= test_config
SRCS := main.c generator.c m_to_csr.c
OBJS := $(SRCS:.c=.o)
TEST_SRCS := test_configurations.c generator.c m_to_csr.c
TEST_OBJS := $(TEST_SRCS:.c=.o)

# # detect OpenMP usage and add flags (best-effort)
# ifeq (, $(filter %#include <omp.h>%,$(shell sed -n '1,200p' $(SRCS) 2>/dev/null)))
# # no-op
# else
# CFLAGS += -fopenmp
# LDLIBS += -fopenmp
# endif

.PHONY: all clean run test

all: $(OUT)

test: $(TEST_OUT)

$(OUT): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_OUT): $(TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# generic rule to compile .c -> .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TEST_OBJS) $(OUT) $(TEST_OUT) test_configurations.o

run: all
	./$(OUT)

#gcc -O3 -Wall -Wextra -march=native -fopenmp -c main.c -o main.o
#for threads in {1..16}; do echo "Testing with $threads threads"; ./executable $threads bcsstk30.mtx; done