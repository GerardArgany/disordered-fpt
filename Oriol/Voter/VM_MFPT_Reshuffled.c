#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../PRNG.h"

static void shuffle_rates(double **rates, int n)
{
    for (int i = 1; i < n - 1; i++) {
        int j = i + rand() / (RAND_MAX / (n - i) + 1);
        double left = rates[i][0];
        double right = rates[i][1];
        rates[i][0] = rates[j][0];
        rates[i][1] = rates[j][1];
        rates[j][0] = left;
        rates[j][1] = right;
    }
}

static int run_trajectory(int n, double **rates, unsigned long max_time)
{
    int index = 1;
    unsigned long time = 0;
    while (index != n) {
        double random_value = genrand_real2();
        if (random_value <= rates[index][0]) {
            index--;
        } else if (random_value <= rates[index][0] + rates[index][1]) {
            index++;
        }
        time++;
        if (index == 0) {
            return -1;
        }
        if (time >= max_time) {
            return -1;
        }
    }
    return (int)time;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s input_file seed\n", argv[0]);
        return EXIT_FAILURE;
    }

    int n, samples_per_shuffle, reshuffles;
    double left_boundary, right_boundary;
    char output_path[512];
    char fpt_output_path[512];
    FILE *input = fopen(argv[1], "r");
    if (input == NULL || fscanf(input, "%d %lf %lf %d %d %511s",
                                &n, &left_boundary, &right_boundary,
                                &samples_per_shuffle, &reshuffles, output_path) != 6) {
        fprintf(stderr, "Could not read input parameters.\n");
        return EXIT_FAILURE;
    }
    fclose(input);

    srand((unsigned)strtoul(argv[2], NULL, 10));
    init_genrand((unsigned long)rand());

    double **rates = malloc((n + 1) * sizeof(*rates));
    for (int i = 0; i <= n; i++) {
        rates[i] = malloc(2 * sizeof(*rates[i]));
        double position = left_boundary + 2.0 * i / n;
        rates[i][0] = 0.25 * (1.0 - position * position);
        rates[i][1] = rates[i][0];
    }

    unsigned long max_time = (unsigned long)pow(n, 4);
    FILE *output = fopen(output_path, "w");
    if (output == NULL) {
        fprintf(stderr, "Could not open output file.\n");
        return EXIT_FAILURE;
    }
    setvbuf(output, NULL, _IOLBF, 0);
    snprintf(fpt_output_path, sizeof(fpt_output_path), "%s_FPT_samples", output_path);
    FILE *fpt_output = fopen(fpt_output_path, "w");
    if (fpt_output == NULL) {
        fprintf(stderr, "Could not open FPT sample output file.\n");
        fclose(output);
        return EXIT_FAILURE;
    }
    setvbuf(fpt_output, NULL, _IOLBF, 0);

    for (int reshuffle = 0; reshuffle < reshuffles; reshuffle++) {
        shuffle_rates(rates, n);
        double sum = 0.0;
        int completed = 0;
        for (int sample = 0; sample < samples_per_shuffle; sample++) {
            int passage_time = run_trajectory(n, rates, max_time);
            if (passage_time >= 0) {
                sum += passage_time / (double)n;
                completed++;
                fprintf(fpt_output, "%d %.12g\n", reshuffle, passage_time / (double)n);
            }
        }
        if (completed > 0) {
            fprintf(output, "%d %.12g %d\n", reshuffle, sum / completed, completed);
        }
    }

    fclose(output);
    fclose(fpt_output);
    for (int i = 0; i <= n; i++) {
        free(rates[i]);
    }
    free(rates);
    return EXIT_SUCCESS;
}