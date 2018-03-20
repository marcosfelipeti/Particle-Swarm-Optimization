#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "svm.h"
#include "time.h"
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>



#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

unsigned int myseed = (unsigned int)time(NULL);
//unsigned int myseed = (unsigned int)10;

Particle g_best;
static double *weights;
double weight;
double mean;
double stdev;
int swarm;
int iterations;

//void print_null(const char *s) {}

void exit_with_help()
{
	printf(
	"Usage: svm-train [options] training_set_file [model_file]\n"
	"options:\n"
	"-s svm_type : set type of SVM (default 0)\n"
	"	0 -- C-SVC		(multi-class classification)\n"
	"	1 -- nu-SVC		(multi-class classification)\n"
	"	2 -- one-class SVM\n"
	"	3 -- epsilon-SVR	(regression)\n"
	"	4 -- nu-SVR		(regression)\n"
	"-t kernel_type : set type of kernel function (default 2)\n"
	"	0 -- linear: u'*v\n"
	"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
	"	4 -- precomputed kernel (kernel values in training_set_file)\n"
	"-d degree : set degree in kernel function (default 3)\n"
	"-g gamma : set gamma in kernel function (default 1/num_features)\n"
	"-r coef0 : set coef0 in kernel function (default 0)\n"
	"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
	"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-m cachesize : set cache memory size in MB (default 100)\n"
	"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
	"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
	"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
	"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
	"-v n: n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void parse_command_line(double v, double c);
void read_problem(const char *filename);
double do_cross_validation();

struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
struct svm_node *x_space;
int cross_validation;
int nr_fold;

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

//check parameters error
void check_p_error()
{
	const char *error_msg;
	error_msg = svm_check_parameter(&prob,&param);
	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}
}

//***********************************************************************************************************************************************************************************************


/*
* PSO time, sweetie
*/

//print one Particle
void print_particle(Particle p) {

	printf("\nValue g: %f\n", p.components[0] ); 
	printf("Value c: %f\n", p.components[1]); 
	printf("P best: %f\n", p.p_best);
	printf("Fitness: %f\n", p.fitness);

	printf("\n\n\n");
}



//print population
void print_swarm(Particle *ps) {

	int i;

	for (i=0; i<swarm; i++) {
		printf("\nIndex: %d\n", i);

		printf("\nValue g: %f\n", ps[i].components[0]); 
		printf("VAlue c: %f\n", ps[i].components[1]); 
		printf("P best: %f\n", ps[i].p_best);
		printf("Fitness: %f\n", ps[i].fitness);

		printf("\n\n\n");

	}
}


//creating random swarm
void initialize_particles(Particle *ps)
	{
		int j;

		for(j = 0; j < swarm; j++){
				double rg = ((float)rand_r(&myseed)/(float)RAND_MAX*(MAXg - MINg));
				double rc = ((float)rand_r(&myseed)/(float)RAND_MAX*(MAXc - MINc));
			
			
				ps[j].components[0] = ps[j].g_value = rg;
				ps[j].components[1] = ps[j].c_value = rc;

				ps[j].p_best = -10000;
		
		}
	}
	

/*initialize random speed*/
void rand_speed(Particle *ps)
	{
		int i, j;
		
		for(j = 0; j < swarm ; j++)
		{
			for (i = 0; i < var_number; i++)
			{
				ps[j].speed[i] = (((float)rand_r(&myseed)/(float)RAND_MAX*(10)));
			}
		}
	}	
	
	
//How good are you?
void evaluate(Particle *ps)
{
	int i;

	for(i=0; i < swarm; i++) {
		parse_command_line(ps[i].g_value, ps[i].c_value);
		check_p_error();
		
			ps[i].fitness = do_cross_validation();

	
		if (ps[i].fitness > ps[i].p_best)
		{
			ps[i].p_best = ps[i].fitness;
			ps[i].components_best_one[0] = ps[i].components[0];
			ps[i].components_best_one[1] = ps[i].components[1];
		}
		
	}
}	

void init_weights()
{
	int i, j;
	weights = (double*)malloc(sizeof(double)*iterations);
	double frac_weight = (sup_weight - inf_weight)/iterations;
	for (i = 0, j = iterations-1; i < iterations; i++,j--)
	{
		weights[j] = inf_weight + i*frac_weight;
	}
}

/*calcular velocidade  da particula*/
void calculate_speed(Particle *ps)
{
	int j;
	
	for(j=0; j < swarm; j++) {
		double r1 = ((float)rand_r(&myseed)/(float)RAND_MAX);
		double r2 = ((float)rand_r(&myseed)/(float)RAND_MAX);

		for (int i = 0; i < var_number; i++)
		{
			ps[j].speed[i] = weight*(ps[j].speed[i] + c1 * r1 *
				(ps[j].components_best_one[i] - ps[j].components[i]) +
				c2 * r2 * (g_best.components[i] - ps[j].components[i]));
		}
	}
}
	
		
/*update positions*/
void update_positions(Particle *ps)
{
	int i, j;
	
	for(j=0; j < swarm; j++) {
		for (i = 0; i < var_number; i++)
		{
			ps[j].components[i] += ps[j].speed[i];
		}
	}
}	



//------------------------------------------------------------------quick sort--------------------------------------------------------
void swap(Particle* a, Particle* b) {

	Particle tmp;

	tmp = *a;
	*a = *b;
	*b = tmp;
}
 
int partition(Particle* vec, int left, int right) {

	int i, j;
 
	i = left;
	for (j = left + 1; j <= right; ++j) {
		if (vec[j].fitness < vec[left].fitness) {
      			++i;
			swap(&vec[i], &vec[j]);
    		}
  	}
  	swap(&vec[left], &vec[i]);
 
	return i;
}
 
void quick_sort(Particle* vec, int left, int right) {

	int r;
 
	if (right > left) {
		r = partition(vec, left, right);
		quick_sort(vec, left, r - 1);
		quick_sort(vec, r + 1, right);
  	}
}

// -------------------------------------------------------end of quick sort----------------------------------------------------------

// somation	
double summation(Particle *ps)
	{
		double sum = 0.0;
		int i;

		for(i = 0; i < swarm; i++)
		{
			sum += ps[i].fitness;
		}

		return sum;
	}
	

	//calculate the mean of the population
void calculate_mean(Particle *ps) {
	double sum = summation(ps);
	mean = sum/swarm;
	printf("\nMean = %f\n", mean);
}


//calculate stdev of the population
void calculate_stdv(Particle *ps) {
	calculate_mean(ps);
	double variance = 0;
	int i;
	
	for (i= 0; i < swarm; i++){
			double difference = pow ((ps[i].fitness - mean), 2);
			variance += difference;
	}
	
	variance = variance/swarm;
	stdev = sqrt(variance);
	
		printf("\nStdev = %f\n\n", stdev);
}
	

//Write in file mean
void write_file_mean( ){

	FILE *m;
   m = fopen("results/mean.txt", "a");
   fprintf(m, "%f\n", mean);
   fclose(m);
}

//Write in file stdev
void write_file_stdev( ){

	FILE *s;
   s = fopen("results/stdev.txt", "a");
   fprintf(s, "%f\n", stdev);
   fclose(s);
}

//Write in file best
void write_file_best( ){

	FILE *b;
   b = fopen("results/best.txt", "a");
   fprintf(b, "%f\n", g_best.fitness);
   fclose(b);
}



//Write in file best individual complete
void write_file_best_complete( ){

	FILE *b;
   b = fopen("results/best_complete.txt", "a");
   fprintf(b, "%f\n", g_best.components[0]);
	fprintf(b, "%f\n", g_best.components[1]);
   fprintf(b, "%f\n", g_best.fitness);
   fprintf(b, "\n\n\n\n");
   fclose(b);
}


//--------------------------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv)
{

	//Create results directory
	struct stat st = {0};
	if (stat("results/", &st) == -1) {
    mkdir("results/", 0700);
	}

	swarm = atoi(argv[1]);
	iterations = atoi(argv[2]);
	
	clock_t startTime = clock();

	int n=0;

	Particle* ps = (Particle*)malloc(swarm*sizeof(Particle));



		/*
	* You're supposed to put the file to train SVM here
	*/
	read_problem("enzymes.txt");

//---------------------------------------------------------------------------------------First swarm-------------------------------------------------------------------
	printf("Creating swarm of particles...\n\n");

	initialize_particles(ps);
	rand_speed(ps);
	evaluate(ps);
	init_weights();
	
	quick_sort(ps, 0, swarm-1);
	
	g_best = ps[swarm-1];
	//print_swarm(ps);
	print_particle(g_best);
	
	calculate_mean(ps);
	write_file_mean();
	calculate_stdv(ps);
	write_file_stdev();
	write_file_best();
	write_file_best_complete( );

	
	

//-------------------------------------------------------------------------repeat for each iteraction----------------------------------------------------------------------------

while ( n < iterations )
{
	printf("\n\n****************************************Generation number= %d*********************************************************************", n+1);
	calculate_speed(ps);
	update_positions(ps);
	evaluate(ps);
	quick_sort(ps, 0, swarm-1);
	g_best = ps[swarm-1];
	print_particle(g_best);
	
	calculate_mean(ps);
	write_file_mean();
	calculate_stdv(ps);
	write_file_stdev();
	write_file_best();
	write_file_best_complete( );
	
	svm_destroy_param(&param);
	
	n++;
}

	printf("Time= %f segundos", double( clock() - startTime ) / (double)CLOCKS_PER_SEC);
	
	// save time in file
	FILE *b;
	b = fopen("best.txt", "a");
	fprintf(b, "%f\n\n", double( clock() - startTime ) / (double)CLOCKS_PER_SEC);
	fclose(b);
   
   
	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);

	return 0;
}

double do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double,prob.l);
	double acc;

	svm_cross_validation(&prob,&param,nr_fold,target);
	
	if(param.svm_type == EPSILON_SVR ||
	   param.svm_type == NU_SVR)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
			((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			);
			free(target);
			return(-1);
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		//printf("Cross Validation Accuracy = %g%%\n\n\n\n",100.0*total_correct/prob.l);
		acc = 100.0*total_correct/prob.l;
		free(target);
		return(acc);
		
	}

}

void parse_command_line(double v, double c)
{
	//void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
    param.gamma = 0.0;     
	param.gamma = v;
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = c;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	cross_validation = 1;
        nr_fold = 10;
	
	//svm_set_print_string_function(print_func);

	// determine filenames


}

// read in a problem (in svmlight format)

void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
}
