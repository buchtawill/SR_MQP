#include <stdio.h>
#include <unistd.h>
#include <time.h>

unsigned long get_jiffies()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long)(ts.tv_sec * sysconf(_SC_CLK_TCK) + ts.tv_nsec / (1e9 / sysconf(_SC_CLK_TCK)));
}

int main()
{
    unsigned long jiffies_per_sec = sysconf(_SC_CLK_TCK);
    if (jiffies_per_sec <= 0) {
        fprintf(stderr, "Failed to determine jiffies per second.\n");
        return 1;
    }

    printf("Jiffies per second: %lu\n", jiffies_per_sec);

    unsigned long start_jiffies = get_jiffies();
    sleep(10);
    unsigned long end_jiffies = get_jiffies();

    unsigned long jiffies_elapsed = end_jiffies - start_jiffies;
    double seconds_elapsed = (double)jiffies_elapsed / jiffies_per_sec;

    printf("Expected sleep time: 10 second\n");
    printf("Jiffies elapsed: %lu\n", jiffies_elapsed);
    printf("Actual time elapsed: %.6f seconds\n", seconds_elapsed);

    double deviation = seconds_elapsed - 10.0;
    printf("Deviation: %.6f seconds\n", deviation);

    return 0;
}