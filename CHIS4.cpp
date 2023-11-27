#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

const int precision = 14;

void readMatrix(const char* filename, double** matrix, int size);
void readVector(const char* filename, double* vector, int size);
void writeMatrixToFile(const char* filename, double** matrix, int size);
void gradientDescent(double** A, double* x, double* b, int size, int& iterations, double& errorNorm, double& residualNorm);

int main() {
    const int size = 10; // Размерность матрицы
    double** A = new double* [size];
    for (int i = 0; i < size; i++) {
        A[i] = new double[size];
    }
    double* x = new double[size];
    double* b = new double[size];

    // Чтение матрицы A из файла
    readMatrix("A.txt", A, size);
    // Чтение вектора x* из файла
    readVector("x_no_error.txt", x, size);
    // Чтение вектора b из файла
    readVector("b.txt", b, size);

    int iterations;
    double errorNorm, residualNorm;

    // Решение СЛАУ методом градиентного спуска
    gradientDescent(A, x, b, size, iterations, errorNorm, residualNorm);

    // Вывод результатов на экран
    std::cout << "Norm of factual error ||x-x*||: " << std::scientific << std::setprecision(precision) << errorNorm << std::endl;
    std::cout << "Norm of residual ||Ax-b||: " << std::scientific << std::setprecision(precision) << residualNorm << std::endl;
    std::cout << "Number of iterations to achieve a given accuracy: " << iterations << std::endl;

    // Запись результатов в файл
    double** results = new double* [3];
    results[0] = new double[3] {errorNorm, 0, 0};
    results[1] = new double[3] {residualNorm, 0, 0};
    results[2] = new double[3] {static_cast<double>(iterations), 0, 0};
    writeMatrixToFile("results.txt", results, 3);

    for (int i = 0; i < size; i++) {
        delete[] A[i];
    }
    delete[] A;
    delete[] x;
    delete[] b;
    delete[] results[0];
    delete[] results[1];
    delete[] results[2];
    delete[] results;

    return 0;
}

void readMatrix(const char* filename, double** matrix, int size) {
    std::ifstream file(filename);
    if (file.is_open()) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                file >> matrix[i][j];
            }
        }
        file.close();
    }
}

void readVector(const char* filename, double* vector, int size) {
    std::ifstream file(filename);
    if (file.is_open()) {
        for (int i = 0; i < size; i++) {
            file >> vector[i];
        }
        file.close();
    }
}

void writeMatrixToFile(const char* filename, double** matrix, int size) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << std::scientific << std::setprecision(precision);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                file << matrix[i][j] << " ";
            }
            file << std::endl;
        }
        file.close();
    }
}

void gradientDescent(double** A, double* x, double* b, int size, int& iterations, double& errorNorm, double& residualNorm) {
    const double epsilon = 1e-10;  // Заданная точность
    const int maxIterations = 1000;  // Максимальное количество итераций
    double alpha = 0.01;  // Размер шага (learning rate)

    // Инициализация вектора градиента
    double* gradient = new double[size];
    // Инициализация вектора невязки
    double* residual = new double[size];

    iterations = 0;

    while (iterations < maxIterations) {
        // Вычисление невязки: Ax - b
        for (int i = 0; i < size; i++) {
            residual[i] = 0;
            for (int j = 0; j < size; j++) {
                residual[i] += A[i][j] * x[j];
            }
            residual[i] -= b[i];
        }

        // Вычисление нормы невязки ||Ax - b||
        residualNorm = 0;
        for (int i = 0; i < size; i++) {
            residualNorm += pow(residual[i], 2);
        }
        residualNorm = sqrt(residualNorm);

        // Проверка условия останова
        if (residualNorm < epsilon) {
            errorNorm = 0;
            for (int i = 0; i < size; i++) {
                errorNorm += pow(x[i], 2);
            }
            errorNorm = sqrt(errorNorm);
            break;
        }

        // Обновление переменных x
        for (int i = 0; i < size; i++) {
            gradient[i] = 0;
            for (int j = 0; j < size; j++) {
                gradient[i] += A[i][j] * residual[j];
            }
            x[i] -= alpha * gradient[i];
        }

        iterations++;
    }

    // Освобождение памяти, выделенной под массивы gradient и residual
    delete[] gradient;
    delete[] residual;
}