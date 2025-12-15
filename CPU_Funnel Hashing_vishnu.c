
//Vishnu Pujari

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <string.h>

#define MAX_LEVELS 100
#define EMPTY_SLOT INT_MIN
#define HASH_TABLE_CAPACITY 10000 // Total number of slots n
#define NUM_INSERTIONS 5000  // Number of keys to insert (n - delta*n)

// Structure representing one Level Ai (array of buckets)
typedef struct {
    int* keys;      // Keys stored in this level
    int* values;    // Corresponding values
    int size;       // Total number of elements (buckets * beta)
} Level;

// Main Funnel Hash Table Structure
typedef struct {
    int capacity;                  // Total number of slots n
    int max_inserts;               // Maximum number of allowed insertions (n - delta*n)
    int num_inserts;               // Current number of insertions done
    int alpha;                     // Number of levels (A1, ..., Aalpha) (alpha = 4*log(1/delta)+10)
    int beta;                      // Size of each subarray (beta = 2*log(1/delta))
    int special_size;              // Size of the special overflow array Aalpha+1
    int primary_size;              // Size of the primary part (divisible by beta)
    int level_count;               // Number of initialized levels
    int level_bucket_counts[MAX_LEVELS];  // Number of buckets at each level
    int level_salts[MAX_LEVELS];           // Random salts per level for hashing
    Level levels[MAX_LEVELS];              // Array of levels (each level has keys and values)
    int special_salt;               // Random salt for hashing into the special array
    int* special_keys;              // Keys in the special overflow array
    int* special_values;            // Values in the special overflow array
} FunnelHashTable;

// Helper: Integer log base 2
int log2_int(int x) { int r = 0; while (x >>= 1) r++; return r; }

// Helper: Generate a random salt
int random_salt() { return rand(); }

// Hash function: combines key and salt
int hash_with_salt(int key, int salt) { return abs((key ^ salt)) & 0x7FFFFFFF; }

// Algorithm 8: Funnel structure initialization
void initialize_funnel_table(FunnelHashTable* table, int capacity, double delta) {
    table->capacity = capacity;
    table->num_inserts = 0;
    table->max_inserts = capacity - (int)(delta * capacity); // n - delta*n

    // Set alpha and beta based on delta
    table->alpha = (int)ceil(4 * log2(1.0 / delta) + 10);
    table->beta = (int)ceil(2 * log2(1.0 / delta));

    // Compute special array size Aalpha+1
    int est_special = (int)(3 * delta * capacity / 4);
    table->special_size = fmax(1, est_special);

    // Primary array size (divisible by beta)
    table->primary_size = (capacity - table->special_size) / table->beta * table->beta;
    table->special_size = capacity - table->primary_size; // Recalculate exact special size



    // Total number of buckets in all levels
    int total_buckets = table->primary_size / table->beta;

    // Initial bucket size for level 1 (based on geometric decrease)
    double a1 = (table->alpha > 0) ? (total_buckets / (4.0 * (1.0 - pow(0.75, table->alpha)))) : total_buckets;

    // Create the levels
    int remaining_buckets = total_buckets, level_count = 0;
    for (int i = 0; i < table->alpha; ++i) {
        int a_i = fmax(1, (int)round(a1 * pow(0.75, i))); // Geometrically decrease by (3/4)^i
        if (remaining_buckets <= 0 || a_i <= 0) break;
        if (a_i > remaining_buckets) a_i = remaining_buckets;

        table->level_bucket_counts[level_count] = a_i;
        table->level_salts[level_count] = random_salt(); // Independent random salt for each level

        // Initialize the level
        Level* lvl = &table->levels[level_count]; // Pointer to current level
        lvl->size = a_i * table->beta; // Size = number of buckets * beta
        lvl->keys = (int*)malloc(lvl->size * sizeof(int)); //Allocate memory for keys
        lvl->values = (int*)malloc(lvl->size * sizeof(int)); //Allocate memory for values
        for (int j = 0; j < lvl->size; ++j) lvl->keys[j] = EMPTY_SLOT; // Initialize keys to EMPTY_SLOT

        remaining_buckets -= a_i; 
        level_count++;
    }

    // If any buckets left, add them to the last level
    if (remaining_buckets > 0 && level_count > 0) { // Only if we have at least one level
        Level* last = &table->levels[level_count - 1]; // Pointer to last level
        int extra = remaining_buckets * table->beta; // Extra size to add
        // Reallocate memory for keys and values in the last level
        last->keys = (int*)realloc(last->keys, (last->size + extra) * sizeof(int));
        last->values = (int*)realloc(last->values, (last->size + extra) * sizeof(int));
        for (int j = last->size; j < last->size + extra; ++j) last->keys[j] = EMPTY_SLOT; // Initialize new slots to EMPTY_SLOT
        // Update size and bucket count for the last level
        last->size += extra;
        table->level_bucket_counts[level_count - 1] += remaining_buckets;
    }

    // Initialize special overflow array (Aalpha+1)
    table->special_keys = (int*)malloc(table->special_size * sizeof(int));
    table->special_values = (int*)malloc(table->special_size * sizeof(int));
    for (int i = 0; i < table->special_size; ++i) table->special_keys[i] = EMPTY_SLOT; // Initialize special keys to EMPTY_SLOT
    // Generate random salt for special array
    table->special_salt = random_salt();
    table->level_count = level_count;
}

// Algorithm 10: Attempted insertion into a level Ai
int attempted_insertion(FunnelHashTable* table, int level_index, int key, int value) { 
    Level* level = &table->levels[level_index]; // Pointer to the level
    // Get number of buckets and compute bucket index for the key
    int num_buckets = table->level_bucket_counts[level_index];
    int bucket = hash_with_salt(key, table->level_salts[level_index]) % num_buckets;
    int start = bucket * table->beta; // Start index for this bucket
    // End index for this bucket
    int end = start + table->beta;

    for (int i = start; i < end; ++i) { // Iterate over the bucket
        if (level->keys[i] == EMPTY_SLOT || level->keys[i] == key) { // Check if slot is empty or key already exists
            // Insert key and value into the level
            level->keys[i] = key;
            level->values[i] = value;
            return 1; // Success
        }
    }
    return 0; // Failed to insert at this level
}

// Algorithm 11: Insert key following funnel strategy
int insert_key(FunnelHashTable* table, int key, int value) { 
    if (table->num_inserts >= table->max_inserts) return 0; // Check if we can insert more keys
    // Try inserting across all levels
    for (int i = 0; i < table->level_count; ++i) { // Iterate over all levels
        // Attempt insertion into the current level
        if (attempted_insertion(table, i, key, value)) {
            table->num_inserts++;
            return 1;
        }
    }

    // Try inserting into the special array (overflow)
    int size = table->special_size;
    int hash = hash_with_salt(key, table->special_salt);
    int probe_limit = fmax(1, (int)ceil(log(log(table->capacity + 1) + 1)));

    for (int j = 0; j < probe_limit; ++j) { // Probing for an empty slot
        // Compute index using double hashing
        int idx = (hash + j) % size;
        if (table->special_keys[idx] == EMPTY_SLOT || table->special_keys[idx] == key) {
            table->special_keys[idx] = key;
            table->special_values[idx] = value;
            table->num_inserts++;
            return 1;
        }
    }

    // Backup 2-slot probing if probing fails
    // Compute two indices for double hashing
    int idx1 = hash % size;
    int idx2 = (hash + 1) % size; 
    if (table->special_keys[idx1] == EMPTY_SLOT) { // Check first index
        table->special_keys[idx1] = key;
        table->special_values[idx1] = value;
        table->num_inserts++;
        return 1;
    } else if (table->special_keys[idx2] == EMPTY_SLOT) { // Check second index
        table->special_keys[idx2] = key;
        table->special_values[idx2] = value;
        table->num_inserts++;
        return 1;
    }

    return 0; // Failed completely
}

// Print funnel hash table structure
void print_table_structure(FunnelHashTable* table) { 
    int total = 0; // Total allocated size
    printf("Funnel Hash Table Levels (capacity = %d):\n", table->capacity); 
    for (int i = 0; i < table->level_count; ++i) { // Iterate over all levels
        int buckets = table->level_bucket_counts[i]; // Get number of buckets for this level
        int elems = buckets * table->beta; // Compute number of elements in this level
        total += elems;
        printf("Level %d: %d buckets × %d = %d elements\n", i + 1, buckets, table->beta, elems);
    }
    printf("Special array size: %d\n", table->special_size);
    total += table->special_size;
    printf("Total allocated size: %d (expected: %d)\n", total, table->capacity);
}

// Print contents of table
void print_table_contents(FunnelHashTable* table) { 
    printf("\n--- Table Contents ---\n");
    for (int i = 0; i < table->level_count; ++i) { 
        Level* lvl = &table->levels[i]; // Pointer to the level
        // Print contents of the current level
        printf("Level %d:\n", i + 1);
        for (int j = 0; j < lvl->size; ++j) {
            if (lvl->keys[j] != EMPTY_SLOT) {
                printf("  [%d] = (key: %d, value: %d)\n", j, lvl->keys[j], lvl->values[j]);
            }
        }
    }
    printf("Special Overflow Array:\n");
    for (int i = 0; i < table->special_size; ++i) {
        if (table->special_keys[i] != EMPTY_SLOT) {
            printf("  [%d] = (key: %d, value: %d)\n", i, table->special_keys[i], table->special_values[i]);
        }
    }
    printf("------------------------\n");
}

// Main function
int main() {
    srand((unsigned int)time(NULL)); // Seed the random number generator

    FunnelHashTable table; 
    int capacity = HASH_TABLE_CAPACITY; 
    double delta = 0.125; // Load factor

    initialize_funnel_table(&table, capacity, delta); // Initialize the funnel hash table
    print_table_structure(&table); 

    clock_t start = clock(); // Start timing

    for (int i = 0; i < NUM_INSERTIONS; ++i) { // Insert keys into the table
        int key = 1000 + i; 
        int value = i; 
        int success = insert_key(&table, key, value); 
        if (!success) printf("Insert key %d: failed\n", key); // Print failure message if insertion fails
    }

    clock_t end = clock(); // End timing

    print_table_contents(&table); 

    double elapsed_time_ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("\nTotal running time: %.3f ms\n", elapsed_time_ms);


    for (int i = 0; i < table.level_count; ++i) { // Free memory allocated for each level
        free(table.levels[i].keys);
        free(table.levels[i].values);
    }
    free(table.special_keys);
    free(table.special_values);

    return 0;
}
