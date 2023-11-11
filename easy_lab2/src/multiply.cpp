// #include <iostream>
// #include <thread>
// #include <vector>
// #include <immintrin.h>
// // #include <omp.h>

// void multiply_block(int start_row, int end_row, int start_col, int end_col, double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P]) {
//     // #pragma omp parallel for
//     for (int i = start_row; i < end_row; i++) {
//         for (int k = 0; k < M; k++) {
//             __m256d vec1 = _mm256_broadcast_sd(&matrix1[i][k]);
//             for (int j = start_col; j < end_col; j += 4) {
//                 // _mm_prefetch(&matrix1[i][k + 1], _MM_HINT_T0);
//                 // _mm_prefetch(&matrix2[k + 1][j], _MM_HINT_T0);
//                 __m256d vec2 = _mm256_loadu_pd(&matrix2[k][j]);
//                 __m256d result = _mm256_mul_pd(vec1, vec2);
//                 __m256d cur_result = _mm256_loadu_pd(&result_matrix[i][j]);
//                 cur_result = _mm256_add_pd(cur_result, result);
//                 _mm256_store_pd(&result_matrix[i][j], cur_result);
//             }
//         }
//     }
// }
// // const int block_size = 128;
// // 32, 196, 
// void matrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P]) {
//     if (N < 128){
//         for (int i = 0; i < N; i++) {
//             for (int k = 0; k < M; k++) {
//                 for (int j = 0; j < P; j++) {
//                     result_matrix[i][j] += matrix1[i][k] * matrix2[k][j];
//                 }
//             }
//         }
//         return;
//     }
//     int block_size = 0;
//     switch (N)
//     {
//     case 512:
//         block_size = 128;
//         break;
//     case 1024:
//         block_size = 256;
//         break;
//     case 2048:
//         block_size = 512;
//         break;
//     case 2560:
//         block_size = 640;
//         break;
//     case 3072:
//         block_size = 768;
//         break;
//     default:
//         break;
//     }
//     std::vector<std::thread> threads;
//     // threads.reserve(N / block_size * P / block_size);
//     int start_row, end_row, start_col, end_col;
//     for (int i = 0; i < N; i += block_size) {
//         for (int j = 0; j < P; j += block_size) {
//             start_row = i;
//             end_row = std::min(i + block_size, N);
//             start_col = j;
//             end_col = std::min(j + block_size, P);
//             threads.emplace_back([start_row, end_row, start_col, end_col, &matrix1, &matrix2, &result_matrix]() {
//                 multiply_block(start_row, end_row, start_col, end_col, matrix1, matrix2, result_matrix);
//             });
//         }
//     }

//     for (std::thread& t : threads) {
//         t.join();
//     }
// }

// // #include <iostream>
// // #include <cstring>

// // // // baseline GFlops = 0.266358
// // // void matrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P]) {
// // // 	for (int i = 0; i < N; i++) {
// // // 		for (int j = 0; j < P; j++) {
// // // 			for (int k = 0; k < M; k++) {
// // // 				result_matrix[i][j] += matrix1[i][k] * matrix2[k][j];
// // // 			}
// // // 		}
// // // 	}
// // // }

// // // // permute GFlops = 0.438253
// // // void matrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P]) {
// // // 	for (int i = 0; i < N; i++) {
// // // 		for (int k = 0; k < M; k++) {
// // // 			for (int j = 0; j < P; j++) {
// // // 				result_matrix[i][j] += matrix1[i][k] * matrix2[k][j];
// // // 			}
// // // 		}
// // // 	}
// // // }

// // // // transposition
// // // void matrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P]) {
// // // 	double matrix2_transposed[P][M];
// // // 	for (int i = 0; i < P; i++) {
// // // 		for (int j = 0; j < M; j++) {
// // // 			matrix2_transposed[i][j] = matrix2[j][i];
// // // 		}
// // // 	}
// // // 	for (int i = 0; i < N; i++) {
// // // 		for (int k = 0; k < M; k++) {
// // // 			for (int j = 0; j < P; j++) {
// // // 				result_matrix[i][j] += matrix1[i][k] * matrix2_transposed[j][k];
// // // 			}
// // // 		}
// // // 	}
// // // }

// // // // vectorization GFlops = 0.528472
// // typedef double vec __attribute__ ((vector_size (64)));
// // vec* alloc(int n){
// // 	vec* p = (vec*) std::aligned_alloc(64, n * 64);
// // 	memset(p, 0, n * 64);
// // 	return p;
// // }

// // // void matrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P]) {
// // // 	int nB = (N + 7) / 8;
// // // 	int mB = (M + 7) / 8;

// // // 	vec* a = alloc(nB * N);
// // // 	vec* b = alloc(mB * M);

// // // 	for (int i = 0; i < N; i++) {
// // // 		for (int j = 0; j < M; j++) {
// // // 			a[i * nB + j / 8][j % 8] = matrix1[i][j]; 
// // // 		}
// // // 	}

// // // 	for (int i = 0; i < M; i++) {
// // // 		for (int j = 0; j < P; j++) {
// // // 			b[i * nB + j / 8][j % 8] = matrix2[j][i];
// // // 		}
// // // 	}

// // // 	for (int i = 0; i < N; i++){
// // // 		for (int j = 0; j < P; j++){
// // // 			vec sum = {};
// // // 			for (int k = 0; k < nB; k++){
// // // 				sum += a[i * nB + k] * b[j * mB + k];
// // // 			}
// // // 			result_matrix[i][j] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
// // // 		}
// // // 	}
// // // }

// // // // Register reuse GFlops = 0.464864
// // // #include <new>

// // // void kernel(double *a, vec *b, vec *c, int x, int y, int l, int r, int n){
// // //     vec t[6][2]{};
// // //     for (int k = l; k < r; k++){
// // //         for (int i = 0; i < 6; i++){
// // //             vec alpha = vec{} + a[(x + i) * n + k];
// // //             for (int j = 0; j < 2; j++){
// // //                 t[i][j] += alpha * b[k * n/8 + y/8 + j];
// // //             }
// // //         }
// // //     }
// // //     for (int i = 0; i < 6; i++){
// // //         for (int j = 0; j < 2; j++){
// // //             c[(x + i) * n/8 + y/8 + j] += t[i][j];
// // //         }
// // //     }
// // // }

// // // void matrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P]) {
// // //     int nx = (N + 5) / 6 * 6;
// // //     int ny = (P + 7) / 8 * 8;

// // //     double *a = (double*) std::aligned_alloc(64, nx * M * sizeof(double));
// // //     vec *b = alloc(M * ny);
// // //     vec *c = alloc(nx * ny);

// // //     for (int i = 0; i < N; i++){
// // //         for (int j = 0; j < M; j++) {
// // //             a[i * M + j] = static_cast<float>(matrix1[i][j]);
// // //         }
// // //     }

// // //     for (int i = 0; i < M; i++){
// // //         for (int j = 0; j < P; j++) {
// // //             reinterpret_cast<double*>(&b[i * ny/8])[j] = matrix2[i][j];
// // //         }
// // //     }

// // //     for (int i = 0; i < nx; i += 6){
// // //         for (int j = 0; j < ny; j += 16){
// // //             kernel(a, b, c, i, j, 0, M, ny);
// // //         }
// // //     }

// // //     for (int i = 0; i < N; i++){
// // //         for (int j = 0; j < P; j++) {
// // //             result_matrix[i][j] = reinterpret_cast<double*>(&c[i * ny/8])[j];
// // //         }
// // //     }

// // //     std::free(a);
// // //     std::free(b);
// // //     std::free(c);
// // // }

// // // // AVX2 GFlops = 1.218697
// // // #include <new>
// // // #include <immintrin.h>

// // // void kernel(double *a, vec *b, vec *c, int x, int y, int l, int r, int n){
// // //     vec t[6][2]{};
    
// // //     // Assuming that 'vec' is a vector of 8 doubles
// // //     for (int k = l; k < r; k++){
// // //         for (int i = 0; i < 6; i++){
// // //             // Broadcast the scalar value a[(x + i) * n + k] across all elements of an AVX-512 register
// // //             vec alpha = _mm512_set1_pd(a[(x + i) * n + k]);
// // //             for (int j = 0; j < 2; j++){
// // //                 // Load the current value of 'b' and 't' into AVX-512 registers
// // //                 vec bv = _mm512_load_pd(&b[k * n/8 + y/8 + j]);
// // //                 vec tv = _mm512_load_pd(&t[i][j]);
                
// // //                 // Perform the FMA operation: tv += alpha * bv
// // //                 vec result = _mm512_fmadd_pd(alpha, bv, tv);
                
// // //                 // Store the result back to 't'
// // //                 _mm512_store_pd(&t[i][j], result);
// // //             }
// // //         }
// // //     }
    
// // //     // Store the results back into 'c'
// // //     for (int i = 0; i < 6; i++){
// // //         for (int j = 0; j < 2; j++){
// // //             vec cv = _mm512_load_pd(&c[(x + i) * n/8 + y/8 + j]);
// // //             vec tv = _mm512_load_pd(&t[i][j]);
            
// // //             // Add the temporary results to 'c'
// // //             vec result = _mm512_add_pd(cv, tv);
            
// // //             // Store the final result back to 'c'
// // //             _mm512_store_pd(&c[(x + i) * n/8 + y/8 + j], result);
// // //         }
// // //     }
// // // }

// // // void matrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P]) {
// // //     int nx = (N + 5) / 6 * 6;
// // //     int ny = (P + 7) / 8 * 8;

// // //     double *a = (double*) std::aligned_alloc(64, nx * M * sizeof(double));
// // //     vec *b = alloc(M * ny);
// // //     vec *c = alloc(nx * ny);

// // //     for (int i = 0; i < N; i++){
// // //         for (int j = 0; j < M; j++) {
// // //             a[i * M + j] = static_cast<float>(matrix1[i][j]);
// // //         }
// // //     }

// // //     for (int i = 0; i < M; i++){
// // //         for (int j = 0; j < P; j++) {
// // //             reinterpret_cast<double*>(&b[i * ny/8])[j] = matrix2[i][j];
// // //         }
// // //     }

// // //     for (int i = 0; i < nx; i += 6){
// // //         for (int j = 0; j < ny; j += 16){
// // //             kernel(a, b, c, i, j, 0, M, ny);
// // //         }
// // //     }

// // //     for (int i = 0; i < N; i++){
// // //         for (int j = 0; j < P; j++) {
// // //             result_matrix[i][j] = reinterpret_cast<double*>(&c[i * ny/8])[j];
// // //         }
// // //     }

// // //     std::free(a);
// // //     std::free(b);
// // //     std::free(c);
// // // }

// // // // Blocking 1.225019
// // // #include <new>
// // // #include <immintrin.h>

// // // void kernel(double *a, vec *b, vec *c, int x, int y, int l, int r, int n){
// // //     vec t[6][2]{};
    
// // //     // Assuming that 'vec' is a vector of 8 doubles
// // //     for (int k = l; k < r; k++){
// // //         for (int i = 0; i < 6; i++){
// // //             // Broadcast the scalar value a[(x + i) * n + k] across all elements of an AVX-512 register
// // //             vec alpha = _mm512_set1_pd(a[(x + i) * n + k]);
// // //             for (int j = 0; j < 2; j++){
// // //                 // Load the current value of 'b' and 't' into AVX-512 registers
// // //                 vec bv = _mm512_load_pd(&b[k * n/8 + y/8 + j]);
// // //                 vec tv = _mm512_load_pd(&t[i][j]);
                
// // //                 // Perform the FMA operation: tv += alpha * bv
// // //                 vec result = _mm512_fmadd_pd(alpha, bv, tv);
                
// // //                 // Store the result back to 't'
// // //                 _mm512_store_pd(&t[i][j], result);
// // //             }
// // //         }
// // //     }
    
// // //     // Store the results back into 'c'
// // //     for (int i = 0; i < 6; i++){
// // //         for (int j = 0; j < 2; j++){
// // //             vec cv = _mm512_load_pd(&c[(x + i) * n/8 + y/8 + j]);
// // //             vec tv = _mm512_load_pd(&t[i][j]);
            
// // //             // Add the temporary results to 'c'
// // //             vec result = _mm512_add_pd(cv, tv);
            
// // //             // Store the final result back to 'c'
// // //             _mm512_store_pd(&c[(x + i) * n/8 + y/8 + j], result);
// // //         }
// // //     }
// // // }

// // // const int s3 = 64;
// // // const int s2 = 120;
// // // const int s1 = 240;

// // // void matrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P]) {
// // //     int nx = (N + 5) / 6 * 6;
// // //     int ny = (P + 7) / 8 * 8;

// // //     double *a = (double*) std::aligned_alloc(64, nx * M * sizeof(double));
// // //     vec *b = alloc(M * ny);
// // //     vec *c = alloc(nx * ny);

// // //     for (int i = 0; i < N; i++){
// // //         for (int j = 0; j < M; j++) {
// // //             a[i * M + j] = static_cast<float>(matrix1[i][j]);
// // //         }
// // //     }

// // //     for (int i = 0; i < M; i++){
// // //         for (int j = 0; j < P; j++) {
// // //             reinterpret_cast<double*>(&b[i * ny/8])[j] = matrix2[i][j];
// // //         }
// // //     }

// // //     // for (int i = 0; i < nx; i += 6){
// // //     //     for (int j = 0; j < ny; j += 16){
// // //     //         kernel(a, b, c, i, j, 0, M, ny);
// // //     //     }
// // //     // }
// // // 	for (int i3 = 0; i3 < ny; i3 += s3) {
// // // 		for (int i2 = 0; i2 < nx; i2 += s2) {
// // // 			for (int i1 = 0; i1 < ny; i1 += s1) {
// // // 				for (int x = i2; x < std::min(i2 + s2, nx); x += 6) {
// // // 					for (int y = i3; y < std::min(i3 + s3, ny); y += 16) {
// // // 						kernel(a, b, c, x, y, i1, std::min(i1 + s1, ny), ny);
// // // 					}
// // // 				}
// // // 			}
// // // 		}
// // // 	}

// // //     for (int i = 0; i < N; i++){
// // //         for (int j = 0; j < P; j++) {
// // //             result_matrix[i][j] = reinterpret_cast<double*>(&c[i * ny/8])[j];
// // //         }
// // //     }

// // //     std::free(a);
// // //     std::free(b);
// // //     std::free(c);
// // // }

// // // openmp GFlops = 3.778442
// // #include <omp.h>
// // #include <new>
// // #include <immintrin.h>

// // void kernel(double *a, vec *b, vec *c, int x, int y, int l, int r, int n){
// //     vec t[6][2]{};
    
// //     // Assuming that 'vec' is a vector of 8 doubles
// //     for (int k = l; k < r; k++){
// //         for (int i = 0; i < 6; i++){
// //             // Broadcast the scalar value a[(x + i) * n + k] across all elements of an AVX-512 register
// //             vec alpha = _mm512_set1_pd(a[(x + i) * n + k]);
// //             for (int j = 0; j < 2; j++){
// //                 // Load the current value of 'b' and 't' into AVX-512 registers
// //                 vec bv = _mm512_load_pd(&b[k * n/8 + y/8 + j]);
// //                 vec tv = _mm512_load_pd(&t[i][j]);
                
// //                 // Perform the FMA operation: tv += alpha * bv
// //                 vec result = _mm512_fmadd_pd(alpha, bv, tv);
                
// //                 // Store the result back to 't'
// //                 _mm512_store_pd(&t[i][j], result);
// //             }
// //         }
// //     }
    
// //     // Store the results back into 'c'
// //     for (int i = 0; i < 6; i++){
// //         for (int j = 0; j < 2; j++){
// //             vec cv = _mm512_load_pd(&c[(x + i) * n/8 + y/8 + j]);
// //             vec tv = _mm512_load_pd(&t[i][j]);
            
// //             // Add the temporary results to 'c'
// //             vec result = _mm512_add_pd(cv, tv);
            
// //             // Store the final result back to 'c'
// //             _mm512_store_pd(&c[(x + i) * n/8 + y/8 + j], result);
// //         }
// //     }
// // }

// // const int s3 = 64;
// // const int s2 = 120;
// // const int s1 = 240;

// // void matrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P]) {
// //     int nx = (N + 5) / 6 * 6;
// //     int ny = (P + 7) / 8 * 8;

// //     double *a = (double*) std::aligned_alloc(64, nx * M * sizeof(double));
// //     vec *b = alloc(M * ny);
// //     vec *c = alloc(nx * ny);

// // 	#pragma omp parallel for collapse(2)
// //     for (int i = 0; i < N; i++){
// //         for (int j = 0; j < M; j++) {
// //             a[i * M + j] = static_cast<float>(matrix1[i][j]);
// //         }
// //     }
// // 	#pragma omp parallel for collapse(2)
// //     for (int i = 0; i < M; i++){
// //         for (int j = 0; j < P; j++) {
// //             reinterpret_cast<double*>(&b[i * ny/8])[j] = matrix2[i][j];
// //         }
// //     }

// //     // for (int i = 0; i < nx; i += 6){
// //     //     for (int j = 0; j < ny; j += 16){
// //     //         kernel(a, b, c, i, j, 0, M, ny);
// //     //     }
// //     // }
// // 	#pragma omp parallel for collapse(3)
// // 	for (int i3 = 0; i3 < ny; i3 += s3) {
// // 		for (int i2 = 0; i2 < nx; i2 += s2) {
// // 			for (int i1 = 0; i1 < ny; i1 += s1) {
// // 				for (int x = i2; x < std::min(i2 + s2, nx); x += 6) {
// // 					for (int y = i3; y < std::min(i3 + s3, ny); y += 16) {
// // 						kernel(a, b, c, x, y, i1, std::min(i1 + s1, ny), ny);
// // 					}
// // 				}
// // 			}
// // 		}
// // 	}
// // 	#pragma omp parallel for collapse(2)
// //     for (int i = 0; i < N; i++){
// //         for (int j = 0; j < P; j++) {
// //             result_matrix[i][j] = reinterpret_cast<double*>(&c[i * ny/8])[j];
// //         }
// //     }

// //     std::free(a);
// //     std::free(b);
// //     std::free(c);
// // }
// #include <immintrin.h>
// #include <omp.h>
// const int BLOCK_SIZE = 64;
// void multiply_block(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P], int row, int col, int inner) {
//     __m512d a, b, c;
//     for (int i = row; i < row + BLOCK_SIZE; ++i) {
//         for (int j = col; j < col + BLOCK_SIZE; j += 8) {
//             c = _mm512_loadu_pd(&result_matrix[i][j]);
//             for (int k = inner; k < inner + BLOCK_SIZE; ++k) {
//                 a = _mm512_set1_pd(matrix1[i][k]);
//                 b = _mm512_loadu_pd(&matrix2[k][j]);
//                 c = _mm512_fmadd_pd(a, b, c);
//             }
//             _mm512_storeu_pd(&result_matrix[i][j], c);
//         }
//     }
// }

// void matrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P]) {
//     if (N < BLOCK_SIZE || M < BLOCK_SIZE || P < BLOCK_SIZE) {
//         for (int i = 0; i < N; ++i) {
//             for (int k = 0; k < M; ++k) {
//                 for (int j = 0; j < P; ++j) {
//                     result_matrix[i][j] += matrix1[i][k] * matrix2[k][j];
//                 }
//             }
//         }
//         return;
//     }
//     // Initialize result matrix to zero
//     for (int i = 0; i < N; ++i)
//         for (int j = 0; j < P; ++j)
//             result_matrix[i][j] = 0;

//     #pragma omp parallel for collapse(2)
//     for (int i = 0; i < N; i += BLOCK_SIZE) {
//         for (int j = 0; j < P; j += BLOCK_SIZE) {
//             for (int k = 0; k < M; k += BLOCK_SIZE) {
//                 multiply_block(matrix1, matrix2, result_matrix, i, j, k);
//             }
//         }
//     }
// }
#include <immintrin.h>
#include <thread>
#include <vector>

const int BLOCK_SIZE = 64;
const int MAX_THREADS = std::thread::hardware_concurrency(); // 获取系统的最大并发线程数

void multiply_block(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P], int row, int col, int inner) {
    __m512d a, b, c;
    for (int i = row; i < row + BLOCK_SIZE; ++i) {
        for (int j = col; j < col + BLOCK_SIZE; j += 8) {
            c = _mm512_loadu_pd(&result_matrix[i][j]);
            for (int k = inner; k < inner + BLOCK_SIZE; ++k) {
                a = _mm512_set1_pd(matrix1[i][k]);
                b = _mm512_loadu_pd(&matrix2[k][j]);
                c = _mm512_fmadd_pd(a, b, c);
            }
            _mm512_storeu_pd(&result_matrix[i][j], c);
        }
    }
}

void matrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P]) {
    if (N < BLOCK_SIZE || M < BLOCK_SIZE || P < BLOCK_SIZE) {
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < M; ++k) {
                for (int j = 0; j < P; ++j) {
                    result_matrix[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
        return;
    }

    // Initialize result matrix to zero
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j)
            result_matrix[i][j] = 0;

    std::vector<std::thread> threads;
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < P; j += BLOCK_SIZE) {
            threads.push_back(std::thread([=, &matrix1, &matrix2, &result_matrix]() {
                for (int k = 0; k < M; k += BLOCK_SIZE) {
                    multiply_block(matrix1, matrix2, result_matrix, i, j, k);
                }
            }));

            // 如果超过最大线程数，等待其中一个完成
            if (threads.size() >= MAX_THREADS) {
                threads.front().join();
                threads.erase(threads.begin());
            }
        }
    }

    // 等待所有线程完成
    for (auto &th : threads) {
        th.join();
    }
}

