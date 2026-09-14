/* Lower-bound macOS timer reference: no Python, sockets, audio, or disk logging.
 * Build: clang -O2 -Wall -Wextra -Werror scripts/native_timer_reference.c \
 *          -o /tmp/stt-native-reference
 * Run: /tmp/stt-native-reference --seconds 10 --repeats 3 > results.json
 *
 * Public sys/event.h: EVFILT_TIMER | NOTE_ABSOLUTE | NOTE_MACHTIME uses Mach
 * absolute ticks. NOTE_CRITICAL with zero leeway minimizes timer coalescing;
 * it does not raise thread priority or promise real-time execution.
 * Output is deferred until every trial has finished. A passing reference is
 * evidence about timer scheduling only, never the full streaming harness.
 */
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/event.h>
#include <unistd.h>

#define MAX_OBSERVATIONS 1000000U
#define SILENCE_FRAMES 50U

struct observation {
    uint64_t started, scheduled, ideal;
};

struct trial {
    double minimum_gap_ms, maximum_gap_ms, maximum_wake_lag_ms;
    double maximum_ideal_lag_ms, actual_span_seconds, ratio;
    bool valid;
};

static mach_timebase_info_data_t timebase;

static uint64_t nanos_to_ticks(uint64_t nanos) {
    __uint128_t top = (__uint128_t)nanos * timebase.denom;
    return (uint64_t)((top + timebase.numer - 1) / timebase.numer);
}

static double ticks_to_seconds(uint64_t ticks) {
    return (double)((long double)ticks * timebase.numer /
                    timebase.denom / 1000000000.0L);
}

static bool parse_seconds(const char *text, double *value) {
    char *end;
    errno = 0;
    double result = strtod(text, &end);
    if (errno || end == text || *end || !isfinite(result) || result < .02 || result > 600)
        return false;
    *value = result;
    return true;
}

static bool parse_repeats(const char *text, unsigned *value) {
    char *end;
    errno = 0;
    long result = strtol(text, &end, 10);
    if (errno || end == text || *end || result < 1 || result > 100)
        return false;
    *value = (unsigned)result;
    return true;
}

static int wait_for_timer(int queue, uint64_t deadline) {
    struct kevent64_s change = {0}, result = {0};
    change.ident = 1;
    change.filter = EVFILT_TIMER;
    change.flags = EV_ADD | EV_ONESHOT;
    change.fflags = NOTE_ABSOLUTE | NOTE_MACHTIME | NOTE_CRITICAL | NOTE_LEEWAY;
    change.data = (int64_t)deadline;
    change.ext[1] = 0;
    uint64_t watchdog = deadline + nanos_to_ticks(1000000000U);
    for (;;) {
        uint64_t now = mach_absolute_time();
        if (now >= watchdog) {
            errno = ETIMEDOUT;
            return -1;
        }
        uint64_t remaining_ns = (uint64_t)ceill(
            (long double)(watchdog - now) * timebase.numer / timebase.denom);
        struct timespec timeout = {
            .tv_sec = (time_t)(remaining_ns / 1000000000U),
            .tv_nsec = (long)(remaining_ns % 1000000000U)
        };
        int count = kevent64(queue, &change, 1, &result, 1, 0, &timeout);
        if (count == -1 && errno == EINTR)
            continue; /* Same absolute deadline: no relative-delay extension. */
        if (count <= 0) {
            if (count == 0) errno = ETIMEDOUT;
            return -1;
        }
        if (result.flags & EV_ERROR) {
            errno = (int)result.data;
            return -1;
        }
        if (result.ident != 1 || result.filter != EVFILT_TIMER) {
            errno = EIO;
            return -1;
        }
        return 0;
    }
}

int main(int argc, char **argv) {
    double seconds = 10;
    unsigned repeats = 3;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--help")) {
            puts("Usage: stt-native-reference [--seconds 0.02..600] [--repeats 1..100]");
            return 0;
        }
        if (!strcmp(argv[i], "--seconds") && i + 1 < argc && parse_seconds(argv[i + 1], &seconds)) {
            ++i;
        } else if (!strcmp(argv[i], "--repeats") && i + 1 < argc && parse_repeats(argv[i + 1], &repeats)) {
            ++i;
        } else {
            fprintf(stderr, "Invalid arguments; use --help.\n");
            return 2;
        }
    }
    size_t speech_frames = (size_t)llround(seconds / .020);
    size_t frames = speech_frames + SILENCE_FRAMES;
    if (frames > MAX_OBSERVATIONS / repeats) {
        fprintf(stderr, "Too many observations; limit is %u.\n", MAX_OBSERVATIONS);
        return 2;
    }
    if (mach_timebase_info(&timebase) != KERN_SUCCESS || !timebase.numer || !timebase.denom) {
        fprintf(stderr, "Cannot determine Mach timebase.\n");
        return 1;
    }
    struct observation *observations = calloc(frames * repeats, sizeof(*observations));
    struct trial *trials = calloc(repeats, sizeof(*trials));
    int queue = -1;
    int exit_status = 1;
    if (!observations || !trials) {
        fprintf(stderr, "Observation allocation failed.\n");
        goto cleanup;
    }
    queue = kqueue();
    if (queue < 0) {
        perror("kqueue");
        goto cleanup;
    }
    uint64_t frame_ticks = nanos_to_ticks(20000000U);
    uint64_t minimum_ticks = nanos_to_ticks(19000000U);
    for (unsigned repeat = 0; repeat < repeats; ++repeat) {
        struct observation *row = observations + repeat * frames;
        uint64_t origin = mach_absolute_time();
        if (origin > INT64_MAX - frames * frame_ticks - nanos_to_ticks(2000000000U)) {
            fprintf(stderr, "Absolute timer deadline overflow.\n");
            goto cleanup;
        }
        uint64_t due = origin + frame_ticks;
        for (size_t index = 0; index < frames; ++index) {
            uint64_t ideal = origin + (index + 1) * frame_ticks;
            if (wait_for_timer(queue, due) < 0) {
                perror("kevent64 timer");
                goto cleanup;
            }
            uint64_t started = mach_absolute_time();
            if (started < due) {
                fprintf(stderr, "Timer returned before its deadline.\n");
                goto cleanup;
            }
            row[index] = (struct observation){started, due, ideal};
            due = ideal + frame_ticks;
            if (started + minimum_ticks > due)
                due = started + minimum_ticks;
        }
        /* Do all statistics after the timed trial, with no intermediate output. */
        struct trial *result = trials + repeat;
        result->minimum_gap_ms = INFINITY;
        for (size_t index = 0; index < frames; ++index) {
            double wake = ticks_to_seconds(row[index].started - row[index].scheduled) * 1000;
            double lag = ticks_to_seconds(row[index].started - row[index].ideal) * 1000;
            if (wake > result->maximum_wake_lag_ms) result->maximum_wake_lag_ms = wake;
            if (lag > result->maximum_ideal_lag_ms) result->maximum_ideal_lag_ms = lag;
            if (index) {
                double gap = ticks_to_seconds(row[index].started - row[index - 1].started) * 1000;
                if (gap > result->maximum_gap_ms) result->maximum_gap_ms = gap;
                if (gap < result->minimum_gap_ms) result->minimum_gap_ms = gap;
            }
        }
        result->actual_span_seconds = ticks_to_seconds(row[frames - 1].started - row[0].started);
        result->ratio = result->actual_span_seconds / ((frames - 1) * .020);
        result->valid = result->minimum_gap_ms >= 18 && result->maximum_gap_ms <= 40 &&
                        result->ratio >= .98 && result->ratio <= 1.02;
    }
    printf("{\"mode\":\"native_timer_lower_bound\",\"clock\":\"mach_absolute_time\","
           "\"timebase_numer\":%u,\"timebase_denom\":%u,\"seconds\":%.9f,\"repeats\":%u,"
           "\"speech_frames\":%zu,\"extra_frames\":%u,\"frames_per_trial\":%zu,\"trials\":[",
           timebase.numer, timebase.denom, seconds, repeats, speech_frames, SILENCE_FRAMES, frames);
    for (unsigned repeat = 0; repeat < repeats; ++repeat) {
        const struct trial *result = trials + repeat;
        printf("%s{\"trial\":%u,\"valid\":%s,\"interval_ms_min\":%.9f,\"interval_ms_max\":%.9f,"
               "\"wakeup_delay_ms_max\":%.9f,\"max_schedule_lag_ms\":%.9f,"
               "\"actual_span_seconds\":%.9f,\"actual_over_ideal\":%.9f,\"observations\":[",
               repeat ? "," : "", repeat + 1, result->valid ? "true" : "false",
               result->minimum_gap_ms, result->maximum_gap_ms, result->maximum_wake_lag_ms,
               result->maximum_ideal_lag_ms, result->actual_span_seconds, result->ratio);
        for (size_t index = 0; index < frames; ++index) {
            const struct observation *row = observations + repeat * frames + index;
            printf("%s{\"index\":%zu,\"started_ticks\":%" PRIu64 ",\"scheduled_ticks\":%" PRIu64
                   ",\"ideal_ticks\":%" PRIu64 "}", index ? "," : "", index,
                   row->started, row->scheduled, row->ideal);
        }
        printf("]}");
    }
    puts("],\"interpretation\":\"Lower-bound native timer scheduling only; no Python, audio transport, provider, or incoming-message validation.\"}");
    exit_status = ferror(stdout) ? 1 : 0;
cleanup:
    if (queue >= 0) close(queue);
    free(trials);
    free(observations);
    return exit_status;
}
