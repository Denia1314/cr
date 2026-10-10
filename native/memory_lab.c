/* Read-only access probe and private child-process fixture. No game parser. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    uint64_t begin;
    uint32_t elixir_milli;
    uint32_t hand[4];
    uint32_t reserved;
    uint64_t end;
} Fixture;

static uint64_t clock_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0;
    return (uint64_t)value.tv_sec * 1000000000ULL + (uint64_t)value.tv_nsec;
}

static uint64_t start_ticks(pid_t pid) {
    char path[64], text[2048];
    snprintf(path, sizeof(path), "/proc/%d/stat", pid);
    FILE *file = fopen(path, "r");
    if (!file) return 0;
    char *line = fgets(text, sizeof(text), file);
    fclose(file);
    if (!line) return 0;
    char *tail = strrchr(text, ')');
    if (!tail) return 0;
    char *save = NULL, *token = strtok_r(tail + 1, " \n", &save);
    for (int field = 3; token; ++field, token = strtok_r(NULL, " \n", &save)) {
        if (field == 22) return strtoull(token, NULL, 10);
    }
    return 0;
}

static bool integer(const char *text, uint64_t *value) {
    if (!text || !*text) return false;
    for (const char *p = text; *p; ++p) if (*p < '0' || *p > '9') return false;
    errno = 0;
    *value = strtoull(text, NULL, 10);
    return !errno;
}

static int probe(pid_t pid, uint64_t expected_start, bool sample_header) {
    uint64_t before = start_ticks(pid);
    if (!before || before != expected_start) {
        puts("{\"schema\":\"memory_lab.probe.v1\",\"identity_matches\":false,\"actual_memory_read\":false}");
        return 2;
    }
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/maps", pid);
    int map_fd = open(path, O_RDONLY | O_CLOEXEC);
    int map_errno = map_fd < 0 ? errno : 0;
    snprintf(path, sizeof(path), "/proc/%d/mem", pid);
    int mem_fd = open(path, O_RDONLY | O_CLOEXEC);
    int mem_errno = mem_fd < 0 ? errno : 0;
    ssize_t header_bytes = 0;
    bool header_verified = false;
    /* Optional bounded platform ELF read, never game state or secrets. */
    if (sample_header && map_fd >= 0 && mem_fd >= 0) {
        FILE *maps = fdopen(dup(map_fd), "r");
        if (maps) {
            char line[2048], permissions[5];
            unsigned count = 0;
            while (++count <= 32768 && fgets(line, sizeof(line), maps)) {
                unsigned long long begin, end, offset;
                int path_at = 0;
                if (sscanf(line, "%llx-%llx %4s %llx %*s %*s %n", &begin, &end, permissions, &offset, &path_at) != 4)
                    continue;
                const char *module = line + path_at;
                bool platform = strncmp(module, "/apex/", 6) == 0 || strncmp(module, "/system/", 8) == 0;
                bool known = strstr(module, "/libc.so") != NULL || strstr(module, "/app_process64") != NULL;
                if (!platform || !known || permissions[0] != 'r' || offset || end <= begin || end - begin < 16 || begin > INT64_MAX)
                    continue;
                unsigned char header[16] = {0};
                header_bytes = pread(mem_fd, header, sizeof(header), (off_t)begin);
                header_verified = header_bytes == 16 && memcmp(header, "\177ELF", 4) == 0 &&
                                  (header[4] == 1 || header[4] == 2) && header[5] == 1;
                break;
            }
            fclose(maps);
        }
    }
    if (map_fd >= 0) close(map_fd);
    if (mem_fd >= 0) close(mem_fd);
    bool matches = before == start_ticks(pid);
    printf("{\"schema\":\"memory_lab.probe.v1\",\"identity_matches\":%s,"
           "\"maps_opened\":%s,\"maps_errno\":%d,\"memory_file_opened\":%s,"
           "\"memory_errno\":%d,\"actual_memory_read\":%s,\"platform_header_verified\":%s,"
           "\"sample_bytes_read\":%zd}\n",
           matches ? "true" : "false", map_fd >= 0 ? "true" : "false", map_errno,
           mem_fd >= 0 ? "true" : "false", mem_errno, header_bytes > 0 ? "true" : "false",
           header_verified ? "true" : "false", header_bytes > 0 ? header_bytes : 0);
    return matches ? 0 : 2;
}

static ssize_t read_child(pid_t pid, void *remote, void *local, size_t length) {
    struct iovec here = { .iov_base = local, .iov_len = length };
    struct iovec there = { .iov_base = remote, .iov_len = length };
    return process_vm_readv(pid, &here, 1, &there, 1, 0);
}

static bool transfer(int fd, void *buffer, size_t length, bool writing) {
    size_t done = 0;
    while (done < length) {
        ssize_t n = writing ? write(fd, (char *)buffer + done, length - done)
                            : read(fd, (char *)buffer + done, length - done);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        done += (size_t)n;
    }
    return true;
}

static int fixture(unsigned samples, unsigned interval_ms) {
    Fixture *remote = mmap(NULL, sizeof(Fixture), PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    long page = sysconf(_SC_PAGESIZE);
    if (remote == MAP_FAILED || page <= 0) return 3;
    char *edge = mmap(NULL, (size_t)page * 2, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (edge == MAP_FAILED || mprotect(edge + page, (size_t)page, PROT_NONE) != 0) return 3;
    int commands[2], acks[2];
    if (pipe(commands) != 0 || pipe(acks) != 0) return 3;
    pid_t child = fork();
    if (child < 0) return 3;
    if (child == 0) {
        close(commands[1]); close(acks[0]);
        uint32_t seq;
        while (transfer(commands[0], &seq, sizeof(seq), false) && seq) {
            remote->begin = (uint64_t)seq * 2 - 1;
            remote->elixir_milli = (seq * 137U) % 10001U;
            for (unsigned slot = 0; slot < 4; ++slot) remote->hand[slot] = (seq + slot) % 8 + 1;
            remote->end = (uint64_t)seq * 2;
            remote->begin = remote->end;
            if (!transfer(acks[1], &seq, sizeof(seq), true)) break;
        }
        _exit(0);
    }
    close(commands[0]); close(acks[1]);
    uint64_t session_clock = clock_ns();
    uint64_t child_start = start_ticks(child);
    unsigned accepted = 0;
    bool failed = !child_start;
    for (uint32_t seq = 1; seq <= samples && !failed; ++seq) {
        uint32_t ack = 0;
        if (!transfer(commands[1], &seq, sizeof(seq), true) ||
            !transfer(acks[0], &ack, sizeof(ack), false) || ack != seq) { failed = true; break; }
        Fixture copy = {0};
        uint64_t begun = clock_ns();
        ssize_t got = read_child(child, remote, &copy, sizeof(copy));
        uint64_t ended = clock_ns();
        bool valid = got == (ssize_t)sizeof(copy) && copy.begin == seq * 2ULL &&
                     copy.end == copy.begin && copy.elixir_milli == (seq * 137U) % 10001U;
        for (unsigned slot = 0; slot < 4; ++slot) valid = valid && copy.hand[slot] == (seq + slot) % 8 + 1;
        valid = valid && child_start == start_ticks(child);
        if (!valid) { failed = true; break; }
        printf("{\"schema\":\"memory_lab.fixture.v1\",\"source\":\"private_child_fixture\","
               "\"session\":\"%d-%" PRIu64 "\",\"sequence\":%u,"
               "\"read_begin_ns\":%" PRIu64 ",\"read_end_ns\":%" PRIu64 ","
               "\"complete\":true,\"consistent\":true,\"elixir_milli\":%u,"
               "\"hand\":[%u,%u,%u,%u]}\n", child, session_clock, seq, begun, ended,
               copy.elixir_milli, copy.hand[0], copy.hand[1], copy.hand[2], copy.hand[3]);
        fflush(stdout);
        ++accepted;
        if (seq < samples) {
            struct timespec delay = { .tv_sec = interval_ms / 1000, .tv_nsec = (long)(interval_ms % 1000) * 1000000L };
            while (nanosleep(&delay, &delay) != 0 && errno == EINTR) {}
        }
    }
    char scratch[32];
    bool bad_address = read_child(child, (void *)(uintptr_t)1, scratch, 1) != 1;
    bool partial = read_child(child, edge + page - 16, scratch, sizeof(scratch)) != (ssize_t)sizeof(scratch);
    uint32_t stop = 0;
    (void)transfer(commands[1], &stop, sizeof(stop), true);
    close(commands[1]); close(acks[0]);
    while (waitpid(child, NULL, 0) < 0 && errno == EINTR) {}
    bool dead = read_child(child, remote, scratch, 1) != 1;
    bool passed = !failed && accepted == samples && bad_address && partial && dead;
    printf("{\"schema\":\"memory_lab.summary.v1\",\"samples\":%u,\"passed\":%s,"
           "\"bad_address_rejected\":%s,\"partial_read_rejected\":%s,\"exited_child_rejected\":%s,"
           "\"game_memory_read\":false}\n", accepted, passed ? "true" : "false",
           bad_address ? "true" : "false", partial ? "true" : "false", dead ? "true" : "false");
    munmap(remote, sizeof(Fixture)); munmap(edge, (size_t)page * 2);
    return passed ? 0 : 3;
}

int main(int argc, char **argv) {
    uint64_t a, b;
    if ((argc == 4 || argc == 5) && integer(argv[2], &a) && integer(argv[3], &b)) {
        bool sample_header = argc == 5 && strcmp(argv[4], "sample-platform-header") == 0;
        if (strcmp(argv[1], "probe") == 0 && a > 0 && a <= INT32_MAX && b > 0 && (argc == 4 || sample_header))
            return probe((pid_t)a, b, sample_header);
        if (argc == 4 && strcmp(argv[1], "fixture") == 0 && a > 0 && a <= 600 && b >= 10 && b <= 1000 && a * b <= 60000)
            return fixture((unsigned)a, (unsigned)b);
    }
    fputs("usage: memory-lab probe PID START_TICKS [sample-platform-header] | fixture SAMPLES INTERVAL_MS\n", stderr);
    return 64;
}
