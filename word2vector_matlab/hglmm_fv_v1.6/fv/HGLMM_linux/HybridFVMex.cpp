#include "mex.h" 
#include "FV.h"

#define NUM_OF_INPUTS 9
#define SAMPLES_INPUT_IDX 0
#define INIT_M_INPUT_IDX 1
#define INIT_S_INPUT_IDX 2
#define INIT_MU_INPUT_IDX 3
#define INIT_SIGMA_INPUT_IDX 4
#define INIT_B_INPUT_IDX 5
#define INIT_PRIOR_INPUT_IDX 6
#define NORM_DESC_INPUT_IDX 7
#define NUM_OF_CORES_INPUT_IDX 8

#define FV_OUTPUT_IDX 0

void mexFunction(int nlhs, mxArray *plhs[], /* Output variables */
	int nrhs, const mxArray *prhs[]) /* Input variables */
{
	if(nrhs != NUM_OF_INPUTS) /* Check the number of arguments */
		mexErrMsgTxt("Wrong number of input arguments.");
	
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
	for (int i=0; i < classes; i++)
	{
		m_for_algo[i] = new fp_type[numOfDims];

		for (int d=0; d < numOfDims; d++)
		{
			m_for_algo[i][d] = mL[d*classes + i];
		}
	}

	s_for_algo = new fp_type*[classes];
	for (int i=0; i < classes; i++)
	{
		s_for_algo[i] = new fp_type[numOfDims];

		for (int d=0; d < numOfDims; d++)
		{
			s_for_algo[i][d] = sL[d*classes + i];
		}
	}

	mu_for_algo = new fp_type*[classes];
	for (int i=0; i < classes; i++)
	{
		mu_for_algo[i] = new fp_type[numOfDims];

		for (int d=0; d < numOfDims; d++)
		{
			mu_for_algo[i][d] = mu[d*classes + i];
		}
	}
	
	sigma_for_algo = new fp_type*[classes];
	for (int i=0; i < classes; i++)
	{
		sigma_for_algo[i] = new fp_type[numOfDims];

		for (int d=0; d < numOfDims; d++)
		{
			sigma_for_algo[i][d] = sigma[d*classes + i];
		}
	}

	b_for_algo = new bool*[classes];
	for (int i=0; i < classes; i++)
	{
		b_for_algo[i] = new bool[numOfDims];

		for (int d=0; d < numOfDims; d++)
		{
			b_for_algo[i][d] = bFlag[d*classes + i];
		}
	}
	
	prior_for_algo = new fp_type[classes];
	
	for (int i=0; i < classes; i++)
	{
		prior_for_algo[i] = priors[i];
	}

	bool* normDesc_ptr =  mxGetLogicals(prhs[NORM_DESC_INPUT_IDX]);
	bool normDesc = normDesc_ptr[0];
	
	double* cores_ptr = mxGetPr(prhs[NUM_OF_CORES_INPUT_IDX]);
	double cores = cores_ptr[0];
	int numOfCores = cores;
	
	
	fp_type* out_fv = new fp_type[2*numOfDims*classes];

	for (int j=0; j < (2*numOfDims*classes); j++)
	{
		out_fv[j]  = 0.0;
	}

	computeFV(numOfSamples, classes, numOfDims, samples_for_algo, m_for_algo, s_for_algo, mu_for_algo, sigma_for_algo, b_for_algo, prior_for_algo, out_fv, normDesc, cores);

	plhs[FV_OUTPUT_IDX] = mxCreateDoubleMatrix(2*numOfDims*classes, 1, mxREAL); 
	double* fvMexOut = mxGetPr(plhs[FV_OUTPUT_IDX]); 

	for (int j=0; j < (2*numOfDims*classes); j++)
	{
		fvMexOut[j]  = out_fv[j];
	}

	delete[] out_fv;

	for (int i=0; i < numOfSamples; i++)
	{
		delete samples_for_algo[i];
	}

	for (int c=0; c < classes; c++)
	{
		delete mu_for_algo[c];
		delete sigma_for_algo[c];
		delete m_for_algo[c];
		delete s_for_algo[c];
		delete b_for_algo[c];
		
	}
	
	delete[] mu_for_algo;
	delete[] sigma_for_algo;
	delete[] m_for_algo;
	delete[] s_for_algo;
	delete[] b_for_algo;
	delete[] samples_for_algo;
	delete[] prior_for_algo;

	return;
}