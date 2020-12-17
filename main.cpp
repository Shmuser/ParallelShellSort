#include <omp.h>
#include <iostream>
#include <time.h>
#include <ctime>
#include <cstdlib>
#include<algorithm>
#include <cmath>
#include <malloc.h>
#include <sys/resource.h>
#include <sstream>

int THREAD_NUM; // Количество потоков
int THREAD_ID; // Номер потока
int DIM_SIZE; // Размерность гиперкуба
#pragma omp threadprivate(THREAD_ID)

void insertion_sort(int *arr, int start, int size, int step) {
    int temp, j;

    for (int k = start + step; k < size; k += step) {
        j = k;
        while (j > start && arr[j - step] > arr[j]) {
            temp = arr[j];
            arr[j] = arr[j - step];
            arr[j - step] = temp;
            j -= step;
        }
    }
}

void check_array(int *arr, int size) {
    for (int i = 0; i < size - 1; ++i) {
        if (arr[i] > arr[i + 1])
            throw "array not sorted";
    }
}

double do_seq_shell(int *arr, int size) {

    double start_time = omp_get_wtime();
    for (int h = size / 2; h > 0; h = h / 2) {
        for (int i = 0; i < h; i++) {
            insertion_sort(arr, i, size, h);
        }
    }
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

double do_par_for_shell(int *arr, int size) {
    double start_time = omp_get_wtime();
    for (int h = size / 2; h > 0; h = h / 2) {
#pragma omp parallel for shared(arr, size, h) schedule(static) num_threads(4)
        for (int i = 0; i < h; i++) {
            insertion_sort(arr, i, size, h);
        }
    }
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

double do_par_sections_shell(int *arr, int size) {
    double start_time = omp_get_wtime();
    for (int h = size / 2; h > 0; h = h / 2) {
        int k = h / 4;
        int s = h % 4;
        int a1 = k + (s > 0);
        int a2 = k + (s > 1);
        int a3 = k + (s > 2);
#pragma omp parallel sections num_threads(4)
        {
#pragma omp section
            {
                for (int i = 0; i < a1; ++i)
                    insertion_sort(arr, i, size, h);
            }
#pragma omp section
            {
                for (int i = a1; i < a1 + a2; ++i)
                    insertion_sort(arr, i, size, h);
            }
#pragma omp section
            {
                for (int i = a1 + a2; i < a1 + a2 + a3; ++i)
                    insertion_sort(arr, i, size, h);
            }
#pragma omp section
            {
                for (int i = a1 + a2 + a3; i < h; ++i)
                    insertion_sort(arr, i, size, h);
            }

        }
    }
    double end_time = omp_get_wtime();
    return end_time - start_time;
}


void init_parall_vars() {
#pragma omp parallel
    {
        THREAD_ID = omp_get_thread_num();
#pragma omp single
        THREAD_NUM = omp_get_num_threads();
    }
    DIM_SIZE = int(log10(double(THREAD_NUM)) /
                   log10(2.0)) + 1;
}


void part_sort(int *arr, int l, int r) {
    int temp[r - l + 1];
    int j = 0;
    for (int i = l; i <= r; i++) {
        temp[j] = arr[i];
        j++;
    }

    std::sort(temp, temp + r - l + 1);

    j = 0;
    for (int i = l; i <= r; i++) {
        arr[i] = temp[j];
        j++;
    }
}

// Функция для определения пар блоков
void initBlockPairs(int *blockPairs, int iterNum) {
    int PairNum = 0, pair;
    bool isExist;
    for (int i = 0; i < 2 * THREAD_NUM; i++) {
        isExist = false;
        for (int j = 0; (j < PairNum) && !isExist; j++)
            if (blockPairs[2 * j + 1] == i)
                isExist = true;
        if (!isExist) {
            pair = i ^ (1 << (DIM_SIZE - iterNum - 1));
            blockPairs[2 * PairNum] = i;
            blockPairs[2 * PairNum + 1] = pair;
            PairNum++;
        }
    }
}

void compareSplitBlocks(int *data, int pFirstBlockStart,
                        int firstBlockSize, int pSecondBlockStart,
                        int secondBlockSize) {
    int totalSize = firstBlockSize + secondBlockSize;
    int *tempBlock = new int[totalSize];
    int i = 0, j = 0, curr = 0;
    while ((i < firstBlockSize) && (j < secondBlockSize)) {
        if (data[pFirstBlockStart + i] < data[pSecondBlockStart + j])
            tempBlock[curr++] = data[pFirstBlockStart + i++];
        else
            tempBlock[curr++] = data[pSecondBlockStart + j++];
    }
    while (i < firstBlockSize)
        tempBlock[curr++] = data[pFirstBlockStart + i++];
    while (j < secondBlockSize)
        tempBlock[curr++] = data[pSecondBlockStart + j++];

    for (int i = 0; i < firstBlockSize; ++i)
        data[pFirstBlockStart + i] = tempBlock[i];
    for (int i = 0; i < secondBlockSize; ++i)
        data[pSecondBlockStart + i] = tempBlock[firstBlockSize + i];

    delete[] tempBlock;
}

bool isSorted(int *arr, int Size) {
    for (int i = 1; (i < Size); i++) {
        if (arr[i] < arr[i - 1])
            return false;
    }
    return true;
}

double parallelShellSort(int *arr, int size) {

    int start_time = omp_get_wtime();
    int *index = new int[2 * THREAD_NUM];
    int *blockSize = new int[2 * THREAD_NUM];
    int *blockPairs = new int[2 * THREAD_NUM];
    for (int i = 0; i < 2 * THREAD_NUM; i++) {
        index[i] = int((i / double(2 * THREAD_NUM)) * size);
        if (i < 2 * THREAD_NUM - 1)
            blockSize[i] = int((i + 1) / double(2 * THREAD_NUM) * size) - index[i];
        else
            blockSize[i] = size - index[i];
    }
    // сортировка изначальных блоков
#pragma omp parallel
    {
        part_sort(arr, index[THREAD_ID * 2],
                  index[THREAD_ID * 2] + blockSize[THREAD_ID * 2] - 1);
        part_sort(arr, index[THREAD_ID * 2 + 1],
                  index[THREAD_ID * 2 + 1] + blockSize[THREAD_ID * 2 + 1] - 1);
    }
    for (int i = 0; (i < DIM_SIZE) &&
                    (!isSorted(arr, size)); i++) {
        initBlockPairs(blockPairs, i);
#pragma omp parallel
        {
            int firstBlock = blockPairs[2 * THREAD_ID];
            int secondBlock = blockPairs[2 * THREAD_ID + 1];
            compareSplitBlocks(arr, index[firstBlock],
                               blockSize[firstBlock], index[secondBlock],
                               blockSize[secondBlock]);
        }
    }

    int i = 1;
    while (!isSorted(arr, size)) {
#pragma omp parallel
        {
            if (i % 2 == 0)
                compareSplitBlocks(arr, index[2 * THREAD_ID],
                                   blockSize[2 * THREAD_ID],
                                   index[2 * THREAD_ID + 1],
                                   blockSize[2 * THREAD_ID + 1]);
            else if (THREAD_ID < THREAD_NUM - 1)
                compareSplitBlocks(arr, index[2 * THREAD_ID + 1],
                                   blockSize[2 * THREAD_ID + 1],
                                   index[2 * THREAD_ID + 2],
                                   blockSize[2 * THREAD_ID + 2]);
        }
        i++;
    }
    delete[] index;
    delete[] blockSize;
    delete[] blockPairs;

    return omp_get_wtime() - start_time;
}


int *gen_array(int size, int seed) {
    int *array = new int[size];
    srand(seed);
    for (int i = 0; i < size; ++i) {
        array[i] = i;
    }
    std::random_shuffle(&array[0], &array[size]);
    return array;
}


int main() {

    init_parall_vars();
    int array_size, test_size;
    std::cout << "Введите размер массива:\n";
    std::cin >> array_size;

    test_size = 10;
    int seed = 41;
    int *array = gen_array(array_size, seed);

    double seq_time = 0, par_for_time = 0, par_sec_time = 0, par_spec_time = 0;

    for (int i = 0; i < test_size; ++i) {
        int *array_copy = new int[array_size];
        std::copy(array, array + array_size, array_copy);
        seq_time += do_seq_shell(array_copy, array_size);
        check_array(array_copy, array_size);
        delete[] array_copy;
    }
    std::cout << seq_time / test_size << " последовательно\n";

//
    for (int i = 0; i < test_size; ++i) {
        int *array_copy = new int[array_size];
        std::copy(array, array + array_size, array_copy);
        par_for_time += do_par_for_shell(array_copy, array_size);
        check_array(array_copy, array_size);
        delete[] array_copy;
    }
    std::cout << par_for_time / test_size << " параллельно for\n";

    for (int i = 0; i < test_size; ++i) {
        int *array_copy = new int[array_size];
        std::copy(array, array + array_size, array_copy);
        par_sec_time += do_par_sections_shell(array_copy, array_size);
        check_array(array_copy, array_size);
        delete[] array_copy;
    }
    std::cout << par_sec_time / test_size << " параллельно sec\n";


    for (int i = 0; i < test_size; ++i) {
        int *array_copy = new int[array_size];
        std::copy(array, array + array_size, array_copy);
        par_spec_time += parallelShellSort(array_copy, array_size);
        check_array(array_copy, array_size);
        delete[] array_copy;
    }
    std::cout << par_spec_time / test_size << " параллельно спец\n";
}
