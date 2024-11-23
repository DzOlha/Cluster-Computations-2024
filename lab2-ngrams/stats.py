import os
import os
import io
import time
import json
from math import ceil
from mpi4py import MPI
from nltk.util import ngrams

MIN_COUNT = 5
N_GRAM = 4

#File for saving the results of program execution to build summary plots later
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.json")

def list_files(start_path):
    files = []
    for root, _, filenames in os.walk(start_path):
        files.extend([os.path.join(root, f) for f in filenames])
    return files

def divide_list(items, num_parts):
    divided = [[] for _ in range(num_parts)]
    for idx, item in enumerate(items):
        divided[idx % num_parts].append(item)
    return divided

def merge_dicts(dict1, dict2):
    for key, value in dict2.items():
        dict1[key] = dict1.get(key, 0) + value
    return dict1

def process_files(file_list, n, min_count):
    ngram_dict = {}
    for file_path in file_list:
        try:
            with io.open(file_path, mode="r", encoding="utf-8") as file:
                for line in file:
                    for ngram in ngrams(line.split(), n):
                        ngram_dict[ngram] = ngram_dict.get(ngram, 0) + 1
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    return {k: v for k, v in ngram_dict.items() if v >= min_count}

def get_most_frequent_ngram(ngram_dict):
    """Find the most frequent n-grams and number of its occurrences"""
    if not ngram_dict:
        return None, 0
    most_frequent = max(ngram_dict.items(), key=lambda x: x[1])
    return most_frequent


def save_results(num_processes, time_taken, most_frequent_ngram, ngram_count, speedup=None):
    # Upload existing results
    if os.path.exists(RESULTS_FILE) and os.path.getsize(RESULTS_FILE) > 0:
        with open(RESULTS_FILE, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in {RESULTS_FILE}, starting with an empty dictionary.")
                results = {}
    else:
        results = {}

    results[str(num_processes)] = {
        "time": time_taken,
        "most_frequent_ngram": most_frequent_ngram,
        "ngram_count": ngram_count,
        "speedup": speedup
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        files = list_files("../../wiki_dataset_statistics/data/fullEnglish")
        file_chunks = divide_list(files, size)
    else:
        file_chunks = None

    local_files = comm.scatter(file_chunks, root=0)
    start_time = time.time()

    local_ngrams = process_files(local_files, N_GRAM, MIN_COUNT)
    all_ngrams = comm.gather(local_ngrams, root=0)

    end_time = time.time()
    local_time = end_time - start_time
    times = comm.gather(local_time, root=0)

    if rank == 0:
        global_ngram_dict = {}
        for ngram_dict in all_ngrams:
            merge_dicts(global_ngram_dict, ngram_dict)

        # Most frequent n-gram
        most_frequent_ngram, ngram_count = get_most_frequent_ngram(global_ngram_dict)

        print(f"Total n-grams: {len(global_ngram_dict)}")
        print(f"Most frequent n-gram: {most_frequent_ngram}, Count: {ngram_count}")

        # Calculate speedup
        time_taken = max(times)
        if size == 1:
            save_results(size, time_taken, most_frequent_ngram, ngram_count, 1)
        else:
            with open(RESULTS_FILE, "r") as f:
                results = json.load(f)
            t1 = results["1"]["time"]
            speedup = t1 / time_taken
            print(f"Speedup with {size} processes: {speedup:.2f}")

            # Save results into json file
            save_results(size, time_taken, most_frequent_ngram, ngram_count, speedup)

if __name__ == "__main__":
    main()
