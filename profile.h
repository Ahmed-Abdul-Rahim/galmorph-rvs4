#ifndef PROFILE_H
#define PROFILE_H

#include <stdint.h>

#if defined(__riscv)

/* ---------------------------------------------------------------------
 * Real target build (cross-compiled with a RISC-V toolchain): read the
 * hardware 'instret' CSR directly. This is what should actually run on
 * the board/simulator for your final numbers.
 * --------------------------------------------------------------------- */

static inline void init_inst_counter(void) {
    /* instret is always counting, nothing to set up */
}

static inline uint64_t get_inst_count(void) {
#if __riscv_xlen == 32
    /* RV32: instret is 64-bit but each CSR read returns only 32 bits.
     * Read low (instret) + high (instreth); re-read high to guard against a
     * low-word rollover between the two reads. A single `rdinstret` here would
     * leave the high 32 bits as garbage. */
    uint32_t lo, hi, hi2;
    do {
        asm volatile("csrr %0, instreth" : "=r"(hi));
        asm volatile("csrr %0, instret"  : "=r"(lo));
        asm volatile("csrr %0, instreth" : "=r"(hi2));
    } while (hi != hi2);
    return ((uint64_t)hi << 32) | lo;
#else
    uint64_t val;
    asm volatile("rdinstret %0" : "=r"(val));
    return val;
#endif
}

#else

/* ---------------------------------------------------------------------
 * Dev host build (plain `gcc`, e.g. x86-64 Ubuntu via the Makefile):
 * rdinstret doesn't exist on this ISA, which is why `make` was failing
 * to assemble profile.h before. There's no single portable "instruction
 * retired" instruction on x86, so this goes through the Linux perf
 * subsystem instead (PERF_COUNT_HW_INSTRUCTIONS), which reads the same
 * kind of PMU counter that instret exposes on RISC-V. The absolute
 * numbers won't match real RISC-V instruction counts -- different ISA,
 * different codegen -- but they're real, per-layer dynamic instruction
 * counts on this machine, so you can compare before/after a patch
 * without needing the RISC-V toolchain/simulator set up yet.
 * --------------------------------------------------------------------- */

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

static int _profile_perf_fd = -1;

static inline void init_inst_counter(void) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(pe);
    pe.config = PERF_COUNT_HW_INSTRUCTIONS;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    _profile_perf_fd = (int)syscall(SYS_perf_event_open, &pe, 0 /* self */, -1 /* any cpu */, -1, 0);

    if (_profile_perf_fd == -1) {
        fprintf(stderr,
            "\n[PROFILE] WARNING: perf_event_open failed (%s).\n"
            "[PROFILE] Instruction counts below will read 0.\n"
            "[PROFILE] This is almost always kernel.perf_event_paranoid blocking\n"
            "[PROFILE] unprivileged counters. Fix with one of:\n"
            "[PROFILE]   sudo sysctl -w kernel.perf_event_paranoid=-1\n"
            "[PROFILE]   sudo ./test_app <sample_prefix>\n\n",
            strerror(errno));
        return;
    }

    ioctl(_profile_perf_fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(_profile_perf_fd, PERF_EVENT_IOC_ENABLE, 0);
}

static inline uint64_t get_inst_count(void) {
    uint64_t count = 0;
    if (_profile_perf_fd == -1) return 0;
    if (read(_profile_perf_fd, &count, sizeof(count)) != (ssize_t)sizeof(count)) return 0;
    return count;
}

#endif

#endif
