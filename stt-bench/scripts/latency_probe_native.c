/* Diagnostic only: timestamp the actual kevent syscall before Python can
 * reacquire its interpreter lock. A bounded optional native observer records
 * this process's main-thread state. No global priorities or settings change. */
#include <errno.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <libproc.h>
#include <stdatomic.h>
#include <stdint.h>
#include <sys/event.h>
#include <sys/resource.h>
#include <unistd.h>
#include <time.h>

struct event_row { uint64_t ident; int32_t filter; };
struct wait_row { double enter, returned, cpu_enter, cpu_returned; int32_t error; };
struct sample_row { double before, after, cpu_seconds; int32_t state, error;
    uint64_t pageins, bytesread, byteswritten; int32_t resource_error; };
struct policy_row { int32_t role, qos, relative_priority, base_priority, current_priority, error; };
static mach_timebase_info_data_t tb;
static double now(void) { return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9; }
static double cpu(void) { struct timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t); return t.tv_sec + t.tv_nsec / 1e9; }
int initialize(void) { return mach_timebase_info(&tb); }
int set_interactive(void) { return pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0); }
void read_policy(struct policy_row *row) {
    task_category_policy_data_t category = {0};
    mach_msg_type_number_t count = TASK_CATEGORY_POLICY_COUNT;
    boolean_t defaults = FALSE;
    row->error = task_policy_get(mach_task_self(), TASK_CATEGORY_POLICY, (task_policy_t)&category, &count, &defaults);
    row->role = category.role;
    qos_class_t qos; int relative;
    int rc = pthread_get_qos_class_np(pthread_self(), &qos, &relative);
    if (rc) row->error = rc;
    row->qos = qos; row->relative_priority = relative;
    thread_extended_info_data_t info = {0};
    count = THREAD_EXTENDED_INFO_COUNT;
    thread_t self = mach_thread_self();
    rc = thread_info(self, THREAD_EXTENDED_INFO, (thread_info_t)&info, &count);
    mach_port_deallocate(mach_task_self(), self);
    if (rc) row->error = rc;
    row->base_priority = info.pth_priority; row->current_priority = info.pth_curpri;
}

int measured_wait(int fd, int maximum, double timeout, struct event_row *out, struct wait_row *row) {
    struct kevent events[1024];
    if (maximum < 1 || maximum > 1024) { row->error = EINVAL; return -1; }
    struct timespec ts = {0};
    if (timeout >= 0) { ts.tv_sec = (time_t)timeout; ts.tv_nsec = (long)((timeout-ts.tv_sec)*1e9); }
    row->cpu_enter = cpu(); row->enter = now();
    int n = kevent(fd, NULL, 0, events, maximum, timeout < 0 ? NULL : &ts);
    int saved_errno = errno;
    row->returned = now(); row->cpu_returned = cpu(); row->error = n < 0 ? saved_errno : 0;
    for (int i=0; i<n; i++) { out[i].ident = events[i].ident; out[i].filter = events[i].filter; }
    return n;
}

static struct sample_row *samples;
static int capacity, used;
static atomic_int active;
static pthread_t observer;
static thread_t target;
static void *observe(void *unused) {
    (void)unused;
    while (atomic_load(&active) && used < capacity) {
        thread_basic_info_data_t info = {0};
        mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
        struct sample_row *r = &samples[used];
        r->before = now();
        r->error = thread_info(target, THREAD_BASIC_INFO, (thread_info_t)&info, &count);
        r->after = now(); r->state = info.run_state;
        r->cpu_seconds = info.user_time.seconds + info.system_time.seconds +
                        (info.user_time.microseconds + info.system_time.microseconds)/1e6;
        struct rusage_info_v2 usage = {0};
        r->resource_error = proc_pid_rusage(getpid(), RUSAGE_INFO_V2, (rusage_info_t *)&usage);
        r->pageins = usage.ri_pageins; r->bytesread = usage.ri_diskio_bytesread;
        r->byteswritten = usage.ri_diskio_byteswritten;
        used++;
        struct timespec delay = {.tv_sec=0, .tv_nsec=1000000};
        nanosleep(&delay, NULL);
    }
    return NULL;
}
int start_observer(struct sample_row *buffer, int count) {
    samples = buffer; capacity = count; used = 0; target = mach_thread_self();
    atomic_store(&active, 1);
    int result = pthread_create(&observer, NULL, observe, NULL);
    if (result) { atomic_store(&active, 0); mach_port_deallocate(mach_task_self(), target); }
    return result;
}
int stop_observer(void) {
    atomic_store(&active, 0); pthread_join(observer, NULL);
    mach_port_deallocate(mach_task_self(), target); return used;
}
