# Makefile for distributed SpMV with MPI and OpenMP
CC ?= /apps/mpich-3.2.1--gcc-9.1.0/bin/mpicc
CFLAGS := -O3 -Wall -Wextra -march=native -fopenmp
LDLIBS := -lm
OUT ?= mtrvec
TEST_OUT ?= test_config
TEST_NUMA_OUT ?= test_config_numa
TEST_MEMORY_OUT ?= test_memory_opt
SRCDIR := src
SRCS := $(SRCDIR)/main.c $(SRCDIR)/generator.c $(SRCDIR)/m_to_csr.c
OBJS := $(SRCS:.c=.o)
TEST_SRCS := $(SRCDIR)/test_configurations.c $(SRCDIR)/generator.c $(SRCDIR)/m_to_csr.c
TEST_OBJS := $(TEST_SRCS:.c=.o)
TEST_NUMA_SRCS := $(SRCDIR)/test_configurations_numa.c $(SRCDIR)/generator.c $(SRCDIR)/m_to_csr.c
TEST_NUMA_OBJS := $(TEST_NUMA_SRCS:.c=.o)
TEST_MEMORY_SRCS := $(SRCDIR)/test_memory_optimizations.c $(SRCDIR)/generator.c $(SRCDIR)/m_to_csr.c
TEST_MEMORY_OBJS := $(TEST_MEMORY_SRCS:.c=.o)

# # detect OpenMP usage and add flags (best-effort)
# ifeq (, $(filter %#include <omp.h>%,$(shell sed -n '1,200p' $(SRCS) 2>/dev/null)))
# # no-op
# else
# CFLAGS += -fopenmp
# LDLIBS += -fopenmp
# endif

.PHONY: all clean run test test_numa test_memory

all: $(OUT)

test: $(TEST_OUT)

test_numa: $(TEST_NUMA_OUT)

test_memory: $(TEST_MEMORY_OUT)

$(OUT): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_OUT): $(TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_NUMA_OUT): $(TEST_NUMA_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_MEMORY_OUT): $(TEST_MEMORY_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# generic rule to compile .c -> .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TEST_OBJS) $(TEST_NUMA_OBJS) $(TEST_MEMORY_OBJS) \
	      $(OUT) $(TEST_OUT) $(TEST_NUMA_OUT) $(TEST_MEMORY_OUT) \
	      $(SRCDIR)/test_configurations.o $(SRCDIR)/test_configurations_numa.o \
	      $(SRCDIR)/test_memory_optimizations.o

run: all
	mpirun -np 2 ./$(OUT) 96 matrices/121k_0p0093.mtx

#gcc -O3 -Wall -Wextra -march=native -fopenmp -c main.c -o main.o
#for threads in {1..16}; do echo "Testing with $threads threads"; ./executable $threads bcsstk30.mtx; done