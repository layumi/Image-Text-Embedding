#include "mex.h" 
#include "Hybrid.h"
#include "Util.h"

#define NUM_OF_INPUTS 11
#define SAMPLES_INPUT_IDX 0
#define INIT_M_INPUT_IDX 1
#define INIT_S_INPUT_IDX 2
#define INIT_MU_INPUT_IDX 3
#define INIT_SIGMA_INPUT_IDX 4
#define INIT_B_INPUT_IDX 5
#define INIT_PRIOR_INPUT_IDX 6
#define MAX_ITERATIONS_INPUT_IDX 7
#define NUMOFCORES_INPUT_IDX 8
#define MINSVALUE_INPUT_IDX 9
#define MINSIGMAVALUE_INPUT_IDX 10


#define M_OUTPUT_IDX 0
#define S_OUTPUT_IDX 1
#define MU_OUTPUT_IDX 2
#define SIGMA_OUTPUT_IDX 3
#define B_OUTPUT_IDX 4
#define PRIORS_OUTPUT_IDX 5
#define SAMPLES_WEIGHTS_OUTPUT_IDX 6

void mexFunction(int nlhs, mxArray *plhs[], /* Output variables */
	int nrhs, const mxArray *prhs[]) /* Input variables */
{
	if(nrhs < (NUM_OF_INPUTS)) /* Check the number of arguments */
		mexErrMsgTxt("Wrong number of input arguments.");
	
	
	double* cores_ptr = mxGetPr(prhs[NUMOFCORES_INPUT_IDX]);
	double cores = cores_ptr[0];
	int numOfCores = cores;
	mexPrintf("Number Of Cores is %d\n", numOfCores); 
	
	double* minS_ptr = mxGetPr(prhs[MINSVALUE_INPUT_IDX]);
	double minS = minS_ptr[0];
	
	double* minSigma_ptr = mxGetPr(prhs[MINSIGMAVALUE_INPUT_IDX]);
	double minSigma = minSigma_ptr[0];
	
	double* samples = mxGetPr(prhs[SAMPLES_INPUT_IDX]);
	int numOfSamples = mxGetM(prhs[SAMPLES_INPUT_IDX]);
	int numOfDims = mxGetN(prhs[SAMPLES_INPUT_IDX]);

	double* mu = mxGetPr(prhs[INIT_MU_INPUT_IDX]);
	int classes = mxGetM(prhs[INIT_MU_INPUT_IDX]);

	int N = mxGetN(prhs[INIT_MU_INPUT_IDX]);
	
	if (N != numOfDims)
	{
		mexErrMsgTxt("Wrong number of dimensions for mu init.");
	}

	double* sigma = mxGetPr(prhs[INIT_SIGMA_INPUT_IDX]);
	int M = mxGetM(prhs[INIT_SIGMA_INPUT_IDX]);
	N = mxGetN(prhs[INIT_SIGMA_INPUT_IDX]);

	if (M != classes)
	{
		mexErrMsgTxt("Wrong number of classes for sigma init.");
	}

	if (N != numOfDims)
	{
		mexErrMsgTxt("Wrong number of dimensions for sigma init.");
	}

	double* mL = mxGetPr(prhs[INIT_M_INPUT_IDX]);
	M = mxGetM(prhs[INIT_M_INPUT_IDX]);
	N = mxGetN(prhs[INIT_M_INPUT_IDX]);

	if (M != classes)
	{
		mexErrMsgTxt("Wrong number of classes for m init.");
	}

	if (N != numOfDims)
	{
		mexErrMsgTxt("Wrong number of dimensions for m init.");
	}

	double* sL = mxGetPr(prhs[INIT_S_INPUT_IDX]);
	M = mxGetM(prhs[INIT_S_INPUT_IDX]);
	N = mxGetN(prhs[INIT_S_INPUT_IDX]);

	if (M != classes)
	{
		mexErrMsgTxt("Wrong number of classes for s init.");
	}

	if (N != numOfDims)
	{
		mexErrMsgTxt("Wrong number of dimensions for s init.");
	}	
	
	double* priors = mxGetPr(prhs[INIT_PRIOR_INPUT_IDX]);
	M = mxGetM(prhs[INIT_PRIOR_INPUT_IDX]);
	
	if (M != classes)
	{
		mexErrMsgTxt("Wrong number of classes for prior init.");
	}

	bool* bFlag = mxGetLogicals(prhs[INIT_B_INPUT_IDX]);
	M = mxGetM(prhs[INIT_B_INPUT_IDX]);
	N = mxGetN(prhs[INIT_B_INPUT_IDX]);

	if (M != classes)
	{
		mexErrMsgTxt("Wrong number of classes for b init.");
	}

	if (N != numOfDims)
	{
		mexErrMsgTxt("Wrong number of dimensions for b init.");
	}
	
	fp_type** samples_for_algo;
	fp_type** m_for_algo;
	fp_type** s_for_algo;
	fp_type** mu_for_algo;
	fp_type** sigma_for_algo;
	bool** b_for_algo;
	fp_type* prior_for_algo;

	fp_type** out_m;
	fp_type** out_s;
	fp_type** out_mu;
	fp_type** out_sigma;
	bool** out_b;
	fp_type* out_priors;

	samples_for_algo = new fp_type*[numOfSamples];
	for (int i=0; i < numOfSamples; i++)
	{
		samples_for_algo[i] = new fp_type[numOfDims];
		for (int d=0; d < numOfDims; d++)
		{
			samples_for_algo[i][d] = samples[d*numOfSamples + i];
		}
	}

	m_for_algo = new fp_type*[classes];
	out_m = new fp_type*[classes];
	for (int i=0; i < classes; i++)
	{
		m_for_algo[i] = new fp_type[numOfDims];
		out_m[i] = new fp_type[numOfDims];

		for (int d=0; d < numOfDims; d++)
		{
			m_for_algo[i][d] = mL[d*classes + i];
			out_m[i][d] = 0.0;
		}
	}

	s_for_algo = new fp_type*[classes];
	out_s = new fp_type*[classes];
	for (int i=0; i < classes; i++)
	{
		s_for_algo[i] = new fp_type[numOfDims];
		out_s[i] = new fp_type[numOfDims];

		for (int d=0; d < numOfDims; d++)
		{
			s_for_algo[i][d] = sL[d*classes + i];
			out_s[i][d] = 0.0;
		}
	}

	mu_for_algo = new fp_type*[classes];
	out_mu = new fp_type*[classes];
	for (int i=0; i < classes; i++)
	{
		mu_for_algo[i] = new fp_type[numOfDims];
		out_mu[i] = new fp_type[numOfDims];

		for (int d=0; d < numOfDims; d++)
		{
			mu_for_algo[i][d] = mu[d*classes + i];
			out_mu[i][d] = 0.0;
		}
	}
	
	sigma_for_algo = new fp_type*[classes];
	out_sigma = new fp_type*[classes];
	for (int i=0; i < classes; i++)
	{
		sigma_for_algo[i] = new fp_type[numOfDims];
		out_sigma[i] = new fp_type[numOfDims];

		for (int d=0; d < numOfDims; d++)
		{
			sigma_for_algo[i][d] = sigma[d*classes + i];
			out_sigma[i][d] = 0.0;
		}
	}

	b_for_algo = new bool*[classes];
	out_b = new bool*[classes];
	for (int i=0; i < classes; i++)
	{
		b_for_algo[i] = new bool[numOfDims];
		out_b[i] = new bool[numOfDims];

		for (int d=0; d < numOfDims; d++)
		{
			b_for_algo[i][d] = bFlag[d*classes + i];
			out_b[i][d] = 0.0;
		}
	}
	
	prior_for_algo = new fp_type[classes];
	out_priors = new fp_type[classes];
	
	for (int i=0; i < classes; i++)
	{
		prior_for_algo[i] = priors[i];
		out_priors[i] = 0.0;
	}

	double* maxNumOfIterations_ptr =  mxGetPr(prhs[MAX_ITERATIONS_INPUT_IDX]);
	int maxNumOfIterations = maxNumOfIterations_ptr[0];

	int outNumOfIterations = 0.0;
	
	fp_type** out_mixture_weights = new fp_type*[numOfSamples];

	for (int i=0; i<numOfSamples; i++)
	{
		out_mixture_weights[i] = new fp_type[classes];
		
		for (int k=0; k < classes; k++)
		{
			out_mixture_weights[i][k] = 0.0;
		}
	}

	lmmEM(numOfSamples, classes, numOfDims, maxNumOfIterations, samples_for_algo, m_for_algo, s_for_algo, mu_for_algo, sigma_for_algo, b_for_algo, prior_for_algo, out_m, out_s, out_mu, out_sigma, out_b, out_priors, &outNumOfIterations, out_mixture_weights, numOfCores, minS, minSigma);
	outNumOfIterations++;

	plhs[SAMPLES_WEIGHTS_OUTPUT_IDX] = mxCreateDoubleMatrix(numOfSamples, classes, mxREAL); 
	double* samplesWeightsMexOut = mxGetPr(plhs[SAMPLES_WEIGHTS_OUTPUT_IDX]); 

	plhs[MU_OUTPUT_IDX] = mxCreateDoubleMatrix(classes, numOfDims, mxREAL); 
	double* MuMexOut = mxGetPr(plhs[MU_OUTPUT_IDX]); 

	plhs[SIGMA_OUTPUT_IDX] = mxCreateDoubleMatrix(classes, numOfDims, mxREAL); 
	double* SigmaMexOut = mxGetPr(plhs[SIGMA_OUTPUT_IDX]); 

	plhs[M_OUTPUT_IDX] = mxCreateDoubleMatrix(classes, numOfDims, mxREAL); 
	double* MMexOut = mxGetPr(plhs[M_OUTPUT_IDX]); 

	plhs[S_OUTPUT_IDX] = mxCreateDoubleMatrix(classes, numOfDims, mxREAL); 
	double* SMexOut = mxGetPr(plhs[S_OUTPUT_IDX]); 

	plhs[B_OUTPUT_IDX] = mxCreateLogicalMatrix (classes, numOfDims); 
	bool* bMexOut = mxGetLogicals(plhs[B_OUTPUT_IDX]); 

	plhs[PRIORS_OUTPUT_IDX] = mxCreateDoubleMatrix(classes, 1, mxREAL); 
	double* priorsMexOut = mxGetPr(plhs[PRIORS_OUTPUT_IDX]); 

	for (int i=0; i < numOfSamples; i++)
	{
		for (int k=0; k < classes; k++)
		{
			samplesWeightsMexOut[k*numOfSamples + i] = out_mixture_weights[i][k];
		}
	}

	for (int k=0; k < classes; k++)
	{
		for (int d=0; d < numOfDims; d++)
		{
			MMexOut[d*classes + k] = out_m[k][d];
			SMexOut[d*classes + k] = out_s[k][d];
			MuMexOut[d*classes + k] = out_mu[k][d];
			SigmaMexOut[d*classes + k] = out_sigma[k][d];
			bMexOut[d*classes + k] = out_b[k][d];
		}

		priorsMexOut[k] = out_priors[k];
	}

	return;
}