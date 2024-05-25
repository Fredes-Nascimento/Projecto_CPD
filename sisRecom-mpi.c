#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ALEATORIO ((double)random() / (double)RAND_MAX)

void preenche_aleatorio_LR(int nU, int nI, int nF, double L[nU][nF], double R[nF][nI]) {
    srandom(0);
    int i, j;
    for(i = 0; i < nU; i++)
        for(j = 0; j < nF; j++)
            L[i][j] = ALEATORIO / nF;
    for(i = 0; i < nF; i++)
        for(j = 0; j < nI; j++)
            R[i][j] = ALEATORIO / nF;
}

void atualiza_LR(int start_row, int end_row, int nI, int nF, double L[][nF], double R[][nI], double B[][nI], double A[][nI], double alpha) {
    int i, j, k;
    double delta;

    for(i = start_row; i < end_row; i++) {
        for(j = 0; j < nI; j++) {
            B[i][j] = 0;
            for(k = 0; k < nF; k++)
                B[i][j] += L[i][k] * R[k][j];
        }
    }

    for(i = start_row; i < end_row; i++) {
        for(j = 0; j < nI; j++) {
            if (A[i][j] != 0) {
                delta = A[i][j] - B[i][j];
                for(k = 0; k < nF; k++) {
                    L[i][k] -= alpha * 2 * delta * (-R[k][j]);
                    R[k][j] -= alpha * 2 * delta * (-L[i][k]);
                }
            }
        }
    }
}

void print_matrix(int nU, int nI, double B[nU][nI], double A[nU][nI]) {
    // Saída - Item recomendado para cada usuário
    int i, j;
    for(i = 0; i < nU; i++) {
        int max_item = -1;
        double max_value = -1;
        for(j = 0; j < nI; j++) {
            if (A[i][j] == 0 && B[i][j] > max_value) {
                max_value = B[i][j];
                max_item = j;
            }
        }
        printf("%d\n", max_item);
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int iteracoes, nU, nI, nF, nDiferentes, i, j;
    double alpha;

    if (argc != 2) {
        if (rank == 0)
            printf("Uso: %s <arquivo_de_entrada>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    FILE *file = fopen(argv[1], "r");
    if (file == NULL) {
        if (rank == 0)
            printf("Erro ao abrir o arquivo %s.\n", argv[1]);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        fscanf(file, "%d", &iteracoes);
        fscanf(file, "%lf", &alpha);
        fscanf(file, "%d", &nF);
        fscanf(file, "%d %d %d", &nU, &nI, &nDiferentes);
    }

    MPI_Bcast(&iteracoes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nF, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nU, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nI, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nDiferentes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double A[nU][nI];
    if (rank == 0) {
        for(i = 0; i < nU; i++) {
            for(j = 0; j < nI; j++) {
                A[i][j] = 0;
            }
        }
        for (i = 0; i < nDiferentes; i++) {
            int row, col;
            double val;
            fscanf(file, "%d %d %lf", &row, &col, &val);
            A[row][col] = val;
        }
        fclose(file);
    }

    MPI_Bcast(A, nU*nI, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double L[nU][nF];
    double B[nU][nI];
    double R[nF][nI];

    preenche_aleatorio_LR(nU, nI, nF, L, R);

    int chunk_size = (nU + size - 1) / size;
    int start_row = rank * chunk_size;
    int end_row = (rank + 1) * chunk_size;
    if (end_row > nU) end_row = nU;

    for (int iter = 0; iter < iteracoes; iter++) {
        atualiza_LR(start_row, end_row, nI, nF, L, R, B, A, alpha);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, L, nF*nU, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, R, nI*nF, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    if (rank == 0)
        print_matrix(nU, nI, B, A);

    MPI_Finalize();
    return 0;
}

