
#define len(x) (sizeof(x) / sizeof(x[0]))
#define iprint(x) printf("%d\n", x)
#define uep printf("uep\n")
#define hop printf("\n")

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "../PRNG.h"
//~ #include "../auxiliarfunctions.c" 


/* Trying to follow https://github.com/MaJerle/c-code-style */

void TransProb_NoisyVoter(double noise_parm, double left_boundary, double right_boundary, 
						  double **transition_probs, int Nl) {

	int i;
	double pos_norm;
	
	if (left_boundary != -right_boundary) {
		printf("left and right boundaries are not symmetric \n");
		printf("left %f and right %f \n", left_boundary, right_boundary);
		exit(EXIT_FAILURE);
	}
	
	if ((right_boundary != 1) || (left_boundary != -1)) {
		printf("In the VM, the boundaries should be at 1 \n");
		exit(EXIT_FAILURE);
	}
		
	for (i = 0; i < Nl + 1; i++) {
		/* No need to set manually BC, L(-1) = 0 and R(1) = 0 naturally */
		pos_norm = left_boundary + 2 * i / (double)(Nl);
		/* left jump prob */
		transition_probs[i][0] = (0.5 * noise_parm * (1 + pos_norm)) + (0.25 * (1 - noise_parm) * (1 - pos_norm * pos_norm)); 
		/* right jump prob */
		transition_probs[i][1] = (0.5 * noise_parm * (1 - pos_norm)) + (0.25 * (1 - noise_parm) * (1 - pos_norm * pos_norm)); 
	}	

	for (i = 0; i < Nl + 1; i++) {
		printf("%d: %g %g\n", i, transition_probs[i][0], transition_probs[i][1]);
	}

	return;
}

void Shuffle_probs(double **array, size_t Nl) {
	/* Arrange the N elements of ARRAY in random order.
	Only effective if N is much smaller than RAND_MAX;
	if this may not be the case, use a better random
	number generator. 
	Adapted from https://stackoverflow.com/questions/6127503/shuffle-array-in-c
	*/
    
    double left_prob, right_prob;
    int k;
    
    if (Nl > 1) {
        size_t i;
        for (i = 0; i < Nl; i++) { /* modified from Nl - 1 to Nl */
          size_t j = i + rand() / (RAND_MAX / ((Nl + 1) - i) + 1); /* modified from Nl to Nl + 1 */
          left_prob = array[j][0];
          right_prob = array[j][1];
          array[j][0] = array[i][0];
          array[j][1] = array[i][1];
          array[i][0] = left_prob;
          array[i][1] = right_prob;
        }
    }
    
    
	for (k = 0; k < Nl + 1; k++) {
		printf("%d: %g %g\n", k, array[k][0], array[k][1]);
	}
	
    return;
}

double position_update(double walker_position, double step_size, int *walker_position_index,
						double left_boundary, double right_boundary,
						double **transition_probs) {
	/* Updates the position of the walker according to the left/right
	 * jumping probabilities **transition_probs.
	 * It does not allow updates if the walker would go further than the
	 * boundaries.
	 */ 
	int new_pos;
	int flag = 0;
	int cn_index;
	double r;
	double cn;

	r = genrand_real2();
	cn = walker_position;
	cn_index = *walker_position_index;
	
	if (r <= transition_probs[cn_index][0]) {
		if (cn != left_boundary) {
			walker_position -= step_size;
			(*walker_position_index)--;
		}
		flag = 1;
	}
	if ((r > transition_probs[cn_index][0]) && (r <= transition_probs[cn_index][0] + transition_probs[cn_index][1])) {
		if (cn != right_boundary) {
			walker_position += step_size;
			(*walker_position_index)++;
		}
		flag = 1;
	}
	if (r > transition_probs[cn_index][0] + transition_probs[cn_index][1]) {
		walker_position = walker_position;
		(*walker_position_index) = (*walker_position_index);
		flag = 1;
	}

	if (flag == 0) {
		printf("wrong position update \n \n");
		exit(EXIT_FAILURE);
	}
	
	//~ printf("\t%d %g\n", (*walker_position_index), walker_position);
	
	return walker_position;
}

int main(int argc, char *argv[]) {
	/*  argv[1] nom de l'arxiu a carregar
	 *  argv[2] numero de la simulacio
	 *  argv[3] seed per al random number 
	 * 
	 * Script that generates a walker at initial_position that will be
	 * absorbed at target_position. We compute the first-passage time to
	 * the target. The domain is bounded between left_boundary and
	 * right_boundary. Time is discrete, recorded in time_counter.
	 * The walker at point i moves to left and right with probabilities 
	 * encoded in the matrix transition_probs[i][0] and transition_probs[i][1]
	 * respectively. Hence, we have space-dependent jumping probabilities.
	 * transition_probs[i][0] + transition_probs[i][1] need not to sum 1,
	 * therefore there can be a probability to stay at i.
	 * 
	 */
	 
    int i, j, k, c, run; /* Indices */
    int flag;
    int intseed; /* Seed of the PRNG */
    int Nl; /* length of the interval -- will be read from input file */
    int nruns; /* number of independent runs -- will be read from input file */
    double initial_position; /* starting position of the walker -- will be read from input file */
    double target_position; /* target position, simulation stops there -- will be read from input file */
    double left_boundary; /* left limit of the boundary of the domain -- will be read from input file */
    double right_boundary; /* right limit of the boundary of the domain -- will be read from input file */
    double noise_parm; /* noise parameter -- will be read from input file */
    double step_size; /* how much the walker advances in a jump */
    double walker_position; /* current position of the walker */
    int walker_position_index; /* index of the current position of the walker: 0 corresponds to left_boundary, Nl corresponds to right_boundary */
    int target_position_index; /* index of the target position of the walker: 0 corresponds to left_boundary, Nl corresponds to right_boundary */
    
    unsigned long time_counter; /* counts num steps by the walker */
    unsigned long time_counter_max; /* max num steps allowed */
    unsigned long fptime; /* first passage time to target_boundary */
	
	double **transition_probs; /* jumping probabilities -- 1st index: discrete position; [i][0] / [i][1]: prob to jump left/right */

    char finput[100], finput_net1[100], fout1[100]; 

    FILE *fp;

	/* PRNG initialization */
	intseed = strtol(argv[2],NULL,10);
	srand((unsigned)time(NULL)*intseed);
	double seed = rand();
	init_genrand(seed);
 
	// Loading params
    sprintf(finput, "%s", argv[1]);
	fp = fopen(finput, "r+");
	rewind(fp);
	fscanf(fp, "%d %lf %lf %lf %lf %lf %d %s", &Nl, &noise_parm, &initial_position, &target_position, 
											&left_boundary, &right_boundary, &nruns, fout1);
	fclose(fp);
	
	printf("%d %lf %lf %lf %lf %lf %d %s\n", Nl, noise_parm, initial_position, target_position, 
											left_boundary, right_boundary, nruns, fout1);

	/* Malloc arrays */
	//~ is_visited = malloc((Nl) * sizeof(*is_visited));
	transition_probs = malloc((Nl + 1) * sizeof(**transition_probs)); // I set Nl + 1 because the jumps are defined from [-1, 1] (inclusion of right boundary)

	for (i = 0; i < Nl + 1; i++) {
		transition_probs[i] = malloc((2) * sizeof(*(transition_probs[i])));
	}

	/* Setting the probability transitions */
	TransProb_NoisyVoter(noise_parm, left_boundary, right_boundary, transition_probs, Nl);

	//~ Shuffle_probs(transition_probs, Nl);

	fp = fopen(fout1, "w+");
	
	//~ nruns = 1;
	/* This will be the basic jump unit -- 2/N because I have a symmetric interval */
	step_size = 2. / (double)(Nl);
	/* index to be used in the jumping probabilities: 0 corresponds to left boundary, Nl corresponds to right_boundary  */
	walker_position_index = round((initial_position - left_boundary) / step_size); 
	target_position_index = round((target_position - left_boundary) / step_size); 
	/* index to be used in the jumping probabilities: 0 corresponds to left boundary, Nl corresponds to right_boundary  */
	time_counter_max = pow(Nl, 5);
		
	for (run = 0; run < nruns; run++) {
		printf("run %d\n", run);
		
		time_counter = 0;
		walker_position = initial_position;
		walker_position_index = round((initial_position - left_boundary) / step_size);
		
		do {
			time_counter++;
			walker_position = position_update(walker_position, step_size, &walker_position_index,
											  left_boundary, right_boundary, transition_probs);
											  
			//~ printf("t=%d; %d %g\n", time_counter, walker_position_index, walker_position);
			
		if (time_counter > time_counter_max) {
			time_counter = -1;
			break;
		}
		
		} while ((time_counter < time_counter_max)
				&&
				(walker_position_index != target_position_index) );
		
		fptime = time_counter;
		fprintf(fp, "%lu %g\n", fptime, walker_position);
		printf("\t pos=%g; time=%lu\n", walker_position, fptime);
	
	}


	/* Freeing - inner level */
	for (i = 0; i < Nl + 1; i++) {
		free(transition_probs[i]);
	}
	/* Freeing - outter level */
	free(transition_probs);
	
	fclose(fp);
	
	remove(finput);

	return 0;
}

















