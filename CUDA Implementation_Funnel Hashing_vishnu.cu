// Vishnu swarup pujari

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <cuda.h>

// Constants for levels and table parameters
#define MAX_LEVELS 100 // Maximum number of levels in the funnel hash table
#define EMPTY_SLOT INT_MIN // Sentinel value for empty slots
#define HASH_TABLE_CAPACITY 10000 // Total capacity of the hash table
#define NUM_INSERTIONS 5000 // Number of key-value pairs to insert

// Macro to catch CUDA errors
#define cudaCheckError() {   cudaError_t e=cudaGetLastError();   if(e!=cudaSuccess) {     printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));     exit(1);   } }

// Struct for each level of the funnel
typedef struct {
    int* keys; 
    int* values;
    int size;
} Level;

// Main Funnel Hash Table structure
typedef struct { 
    int capacity; 
    int max_inserts; 
    int alpha; // Number of levels in the funnel
    int beta; // Number of slots per bucket 
    int special_size; // Size of the special overflow array
    int primary_size; // Size of the primary array
    int level_count; // Number of levels actually used
    int level_bucket_counts[MAX_LEVELS]; 
    int level_salts[MAX_LEVELS]; // Random salts for each level
    Level levels[MAX_LEVELS];   // Array of levels
    int special_salt; // Salt for the special overflow array
    int* special_keys; // Keys for the special overflow array
    int* special_values; // Values for the special overflow array
} FunnelHashTable;

// Simple salted hash function used to spread keys
__device__ int hash_with_salt(int key, int salt) {
    return abs((key ^ salt)) & 0x7FFFFFFF; // Ensure non-negative hash
}

// Kernel to initialize array memory on device to EMPTY_SLOT
__global__ void initialize_memory(int* arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global index
    if (idx < n) arr[idx] = EMPTY_SLOT; // Set to EMPTY_SLOT if within bounds
}

// Algorithm 10: Attempted Insertion
__device__ int attempted_insertion(FunnelHashTable* table, int level_idx, int key, int value) {
    Level* level = &table->levels[level_idx];
    int num_buckets = table->level_bucket_counts[level_idx];
    int bucket = hash_with_salt(key, table->level_salts[level_idx]) % num_buckets;
    int start = bucket * table->beta;
    int end = start + table->beta;

    for (int i = start; i < end; i++) {
        int old = atomicCAS(&level->keys[i], EMPTY_SLOT, key);
        if (old == EMPTY_SLOT || old == key) {
            level->values[i] = value;
            return 1;
        }
    }
    return 0;
}

// Algorithm 11: Insert Key into Funnel Hash Table
__device__ int insert_key(FunnelHashTable* table, int key, int value) {
    for (int i = 0; i < table->level_count; i++) {
        if (attempted_insertion(table, i, key, value)) return 1;
    }

    // Fallback: Insert into overflow array (A_{alpha+1})
    int size = table->special_size;
    int hash = hash_with_salt(key, table->special_salt);
    int probe_limit = max(1, (int)ceil(log(log((double)(table->capacity + 1)) + 1)));

    for (int j = 0; j < probe_limit; j++) {
        int idx = (hash + j) % size;
        int old = atomicCAS(&table->special_keys[idx], EMPTY_SLOT, key);
        if (old == EMPTY_SLOT || old == key) {
            table->special_values[idx] = value;
            return 1;
        }
    }

    // Last-chance fallback to two alternative indices
    int idx1 = hash % size;
    int idx2 = (hash + 1) % size;
    if (atomicCAS(&table->special_keys[idx1], EMPTY_SLOT, key) == EMPTY_SLOT) {
        table->special_values[idx1] = value;
        return 1;
    } else if (atomicCAS(&table->special_keys[idx2], EMPTY_SLOT, key) == EMPTY_SLOT) {
        table->special_values[idx2] = value;
        return 1;
    }

    return 0;
}

// CUDA kernel to insert all key-value pairs in parallel
__global__ void insert_keys_kernel(FunnelHashTable* table, int* keys, int* values, int n) {
    extern __shared__ int shared[];
    int* shared_keys = shared; // Shared memory for keys
    int* shared_values = &shared[blockDim.x]; // Shared memory for values

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global index
    // Load keys and values into shared memory
    if (idx < n) {
        shared_keys[threadIdx.x] = keys[idx];
        shared_values[threadIdx.x] = values[idx];
    }
    __syncthreads();

    if (idx < n) {
        insert_key(table, shared_keys[threadIdx.x], shared_values[threadIdx.x]);
    }
}

// Algorithm 8: Split array into levels (initialization of structure)
void initialize_table(FunnelHashTable* table, int capacity, double delta) {
    table->capacity = capacity;
    table->max_inserts = capacity - (int)(delta * capacity);
    table->alpha = (int)ceil(4 * log2(1.0 / delta) + 10);
    table->beta = (int)ceil(2 * log2(1.0 / delta));

    int est_special = (int)(3 * delta * capacity / 4);
    table->special_size = fmax(1, est_special);
    table->primary_size = (capacity - table->special_size) / table->beta * table->beta;
    table->special_size = capacity - table->primary_size;

    int total_buckets = table->primary_size / table->beta;
    double a1 = (table->alpha > 0) ? (total_buckets / (4.0 * (1.0 - pow(0.75, table->alpha)))) : total_buckets;

    int remaining_buckets = total_buckets;
    int level_count = 0;
    for (int i = 0; i < table->alpha; i++) {
        int a_i = fmax(1, (int)round(a1 * pow(0.75, i)));
        if (remaining_buckets <= 0 || a_i <= 0) break;
        if (a_i > remaining_buckets) a_i = remaining_buckets;

        table->level_bucket_counts[level_count] = a_i;
        table->level_salts[level_count] = rand();
        table->levels[level_count].size = a_i * table->beta;
        remaining_buckets -= a_i;
        level_count++;
    }
    table->level_count = level_count;
    table->special_salt = rand();
}

// Utility: Print structure of table levels
void print_table_structure(FunnelHashTable* table) {
    int total = 0;
    printf("Funnel Hash Table Levels (capacity = %d):\n", table->capacity);
    for (int i = 0; i < table->level_count; i++) {
        int buckets = table->level_bucket_counts[i];
        int elems = buckets * table->beta;
        total += elems;
        printf("Level %d: %d buckets × %d = %d elements\n", i+1, buckets, table->beta, elems);
    }
    printf("Special array size: %d\n", table->special_size);
    total += table->special_size;
    printf("Total allocated size: %d (expected: %d)\n", total, table->capacity);
}

// Utility: Print final table contents
void print_table_contents(FunnelHashTable* table) {
    printf("\n--- Table Contents ---\n");
    for (int i = 0; i < table->level_count; i++) {
        printf("Level %d:\n", i + 1);
        int sz = table->levels[i].size;
        int* h_keys = (int*)malloc(sz * sizeof(int));
        int* h_values = (int*)malloc(sz * sizeof(int));
        cudaMemcpy(h_keys, table->levels[i].keys, sz * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_values, table->levels[i].values, sz * sizeof(int), cudaMemcpyDeviceToHost);
        for (int j = 0; j < sz; j++) {
            if (h_keys[j] != EMPTY_SLOT) {
                printf("  [%d] = (key: %d, value: %d)\n", j, h_keys[j], h_values[j]);
            }
        }
        free(h_keys);
        free(h_values);
    }

    printf("Special Overflow Array:\n");
    int* h_special_keys = (int*)malloc(table->special_size * sizeof(int));
    int* h_special_values = (int*)malloc(table->special_size * sizeof(int));
    cudaMemcpy(h_special_keys, table->special_keys, table->special_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_special_values, table->special_values, table->special_size * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < table->special_size; i++) {
        if (h_special_keys[i] != EMPTY_SLOT) {
            printf("  [%d] = (key: %d, value: %d)\n", i, h_special_keys[i], h_special_values[i]);
        }
    }
    free(h_special_keys);
    free(h_special_values);
}

// Main function
int main() {
    FunnelHashTable h_table;
    initialize_table(&h_table, HASH_TABLE_CAPACITY, 0.125);  // Algorithm 8

    // Allocate GPU memory for all levels and special array
    for (int i = 0; i < h_table.level_count; i++) {
        int sz = h_table.levels[i].size;
        cudaMalloc(&(h_table.levels[i].keys), sizeof(int) * sz);
        cudaMalloc(&(h_table.levels[i].values), sizeof(int) * sz);
    }
    cudaMalloc(&(h_table.special_keys), sizeof(int) * h_table.special_size);
    cudaMalloc(&(h_table.special_values), sizeof(int) * h_table.special_size);
    cudaCheckError();

    // Initialize device memory
    for (int i = 0; i < h_table.level_count; i++) {
        int sz = h_table.levels[i].size;
        initialize_memory<<<(sz+255)/256, 256>>>(h_table.levels[i].keys, sz);
    }
    initialize_memory<<<(h_table.special_size+255)/256, 256>>>(h_table.special_keys, h_table.special_size);
    cudaDeviceSynchronize();
    cudaCheckError();

    print_table_structure(&h_table);

    // Generate input data
    int* h_keys = (int*)malloc(NUM_INSERTIONS * sizeof(int));
    int* h_values = (int*)malloc(NUM_INSERTIONS * sizeof(int));
    for (int i = 0; i < NUM_INSERTIONS; i++) {
        h_keys[i] = 1000 + i;
        h_values[i] = i;
    }

    int* d_keys, *d_values;
    cudaMalloc(&d_keys, NUM_INSERTIONS * sizeof(int));
    cudaMalloc(&d_values, NUM_INSERTIONS * sizeof(int));
    cudaMemcpy(d_keys, h_keys, NUM_INSERTIONS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, NUM_INSERTIONS * sizeof(int), cudaMemcpyHostToDevice);

    FunnelHashTable* d_table;
    cudaMalloc(&d_table, sizeof(FunnelHashTable));
    cudaMemcpy(d_table, &h_table, sizeof(FunnelHashTable), cudaMemcpyHostToDevice);
    cudaCheckError();

    // Time the GPU execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    size_t sharedMemSize = 2 * 256 * sizeof(int);  // Shared memory for keys and values
    insert_keys_kernel<<<(NUM_INSERTIONS + 255) / 256, 256, sharedMemSize>>>(d_table, d_keys, d_values, NUM_INSERTIONS);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();
    cudaCheckError();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    print_table_contents(&h_table);
    printf("Total running time: %.3f ms\n", milliseconds);

    // Free device and host memory
    for (int i = 0; i < h_table.level_count; i++) {
        cudaFree(h_table.levels[i].keys);
        cudaFree(h_table.levels[i].values);
    }
    cudaFree(h_table.special_keys);
    cudaFree(h_table.special_values);
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_table);
    free(h_keys);
    free(h_values);

    return 0;
}
