/* File: mpi_vector_ops.c
 * COMP 137-1 Spring 2020
 */
#include <stdio.h>
#include <stdlib.h>
char* infile = NULL;
char* outfile = NULL;

int readInputFile(char* filename, long* n_p, double* x_p, double** A_p, double** B_p)
{
    long i;
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) return 0;
    fscanf(fp, "%ld\n", n_p);
    fscanf(fp, "%lf\n", x_p);
    *A_p = (double *)malloc(*n_p*sizeof(double));
    *B_p = (double *)malloc(*n_p*sizeof(double));
    for (i=0; i<*n_p; i++) fscanf(fp, "%lf\n", (*A_p)+i);
    for (i=0; i<*n_p; i++) fscanf(fp, "%lf\n", (*B_p)+i);
    return 1;
}

__global__ void cudaSolution(double number, double* W, double* X, double* Y, double* Z) {

        Y[threadIdx.x] = number*W[threadIdx.x];
        Z[threadIdx.x] = W[threadIdx.x]*X[threadIdx.x];
        
}        


int writeOutputFile(char* filename, long n,  double* C, double* D)
{
    long i;
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) return 0;
    fprintf(fp, "%ld\n", n);
    for (i=0; i<n; i++) fprintf(fp, "%lf\n", C[i]);
    for (i=0; i<n; i++) fprintf(fp, "%lf\n", D[i]);
    return 1;
}



int main(int argc, char* argv[])
{
    long    n=0;     /* size of input arrays */
    double  x;       /* input scalar */
    double* A;       /* input vector */
    double* B;       /* input vector */
    double* C;       /* output vector xA */
    double* D;       /* output vector A*B */
    double* E_D;
    double* F_D;
    double* G_D; 
    double* H_D; 
    

    /* read input data */
    if (argc<3)
    {
        n = -1;
        fprintf(stderr, "Command line arguments are required.\n");
        fprintf(stderr, "argv[1] = name of input file\n");
        fprintf(stderr, "argv[2] = name of input file\n");
    }
    else
    {
        infile = argv[1];
        outfile = argv[2];
        if (!readInputFile(infile, &n, &x, &A, &B))
        {
            fprintf(stderr, "Error opening input files. Aborting.\n");
            n = -1;
        }
    }
    
    double sizesGrid = 1;
    double sizesBlock = 1024;

       if (n > sizesBlock) {
         sizesGrid = (int)ceil((float) n/sizesBlock);
       }
         int sizing = n * sizeof(double);

       cudaMalloc(&E_D, sizing);
       cudaMalloc(&F_D, sizing);
       cudaMalloc(&G_D, sizing);
       cudaMalloc(&H_D, sizing);

       cudaMemcpy(E_D, A, sizing, cudaMemcpyHostToDevice);
       cudaMemcpy(F_D, B, sizing, cudaMemcpyHostToDevice);

       cudaSolution<<<sizesGrid,sizesBlock>>>(x, E_D, F_D, G_D, H_D); 
       C = (double *)malloc(sizing * sizeof(double));
       D = (double *)malloc(sizing * sizeof(double));

      cudaMemcpy(C, G_D, sizing, cudaMemcpyDeviceToHost);
      cudaMemcpy(D, H_D, sizing, cudaMemcpyDeviceToHost);

    if (n < 0)
    {
        fprintf(stderr, "Aborting task due to input errors.\n");
        exit(1);
    }

    cudaDeviceSynchronize();

    if (!writeOutputFile(outfile, n, C, D))
    {
        fprintf(stderr, "Error opening output file. Aborting.\n");
        exit(1);
    }
    
/* free all dynamic memory allocation */
    free(A);
    free(B);
    free(C);
    free(D);
    cudaFree(E_D);
    cudaFree(F_D);
    cudaFree(G_D);
    cudaFree(H_D);

    cudaDeviceSynchronize();
 
    return 0;
}
