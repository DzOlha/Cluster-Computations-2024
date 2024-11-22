import os
import re
import time
from mpi4py import MPI
import argparse
import random
import numpy as np
from scipy.optimize import least_squares


# Define a function to calculate execution time based on the parameters
def execution_time_for_params(granularity, broadcast_rate, num_processes):
    """
    Function that calculates execution time for the given combination of parameters
    This will call the `parallel_text_processing` method for each combination.
    """
    # Placeholder for calling parallel processing function
    execution_time = parallel_text_processing(granularity, broadcast_rate)
    return execution_time


# --- Monte Carlo Simulation ---
def monte_carlo_simulation(num_simulations, granularity_range, broadcast_rate_range, num_processes_range):
    """
    Perform Monte Carlo simulations to explore the parameter space.
    """
    data = []

    for _ in range(num_simulations):
        granularity = random.randint(*granularity_range)
        broadcast_rate = random.randint(*broadcast_rate_range)
        num_processes = random.randint(*num_processes_range)

        # Run parallel processing with these parameters
        execution_time = execution_time_for_params(granularity, broadcast_rate, num_processes)

        # Append the results including execution time
        data.append((granularity, broadcast_rate, num_processes, execution_time))

    return np.array(data)


# --- Least Squares Optimization ---
def least_squares_optimization(data):
    """
    Perform Least Squares Optimization to minimize the execution time.
    """

    def residuals(params, data):
        """
        Calculate the residuals for Least Squares optimization
        params: [granularity, broadcast_rate, num_processes]
        """
        granularity, broadcast_rate, num_processes = params
        predictions = np.array([execution_time_for_params(granularity, broadcast_rate, num_processes) for
                                granularity, broadcast_rate, num_processes, _ in data])
        actual = data[:, 3]  # The actual execution times
        return predictions - actual

    # Initial guess for the parameters
    initial_guess = np.mean(data[:, :3], axis=0)  # Mean of the existing values as initial guess

    # Perform least squares optimization
    result = least_squares(residuals, initial_guess, args=(data,))

    # The optimized parameters
    optimized_params = result.x
    return optimized_params


# --- Combined Optimization ---
def optimize_parameters(monte_carlo_simulations=100, granularity_range=(1, 5), broadcast_rate_range=(1, 10),
                        num_processes_range=(1, 8)):
    """
    Combine Monte Carlo and Least Squares optimization to find the best combination of parameters.
    """
    # Step 1: Run Monte Carlo simulation to generate a large set of parameter combinations
    data = monte_carlo_simulation(monte_carlo_simulations, granularity_range, broadcast_rate_range, num_processes_range)

    # Step 2: Use Least Squares optimization to find the best parameter combination
    optimized_params = least_squares_optimization(data)

    return optimized_params


# --- Command-Line Argument Parsing ---
def parse_arguments():
    """
    Parse command-line arguments to get granularity and broadcast rate.
    """
    parser = argparse.ArgumentParser(description='Parallel texts processing app')
    parser.add_argument('--gran', action="store", dest='granularity', default=1)
    parser.add_argument('--bcast_rate', action="store", dest='broadcast_rate', default=1)
    return parser.parse_args()


# --- File Listing Function ---
def list_files_walk(start_path='.'):
    """
    Recursively list all files in a directory and its subdirectories.
    """
    files = []
    for root, dirs, files_in_dir in os.walk(start_path):
        for file in files_in_dir:
            files.append(os.path.join(root, file))
    return files


# --- Divide Files Between MPI Processes ---
def distribute_files_across_processes(filenames, processes_count):
    """
    Distribute the list of files across multiple MPI processes.
    """
    files_per_process = []
    for i in range(processes_count):
        files_for_process = []
        curr_filename_index = i
        while curr_filename_index < len(filenames):
            files_for_process.append(filenames[curr_filename_index])
            curr_filename_index += processes_count
        files_per_process.append(files_for_process)
    return files_per_process


# --- Merging Dictionaries ---
def merge_word_counts(dict1, dict2):
    """
    Merge two dictionaries containing word counts. If a word exists in both, sum the counts.
    """
    for word, count in dict2.items():
        dict1[word] = dict1.get(word, 0) + count
    return dict1


# --- Process File to Count Word Frequency ---
def process_file_for_word_counts(file_path):
    """
    Process a file to count the frequency of each word, ignoring punctuation and case.
    """
    word_count = {}
    with open(file_path, 'r') as file:
        for line in file:
            for word in re.split(r"[ ,\.\(\)\;\|\"\d\\\/\+\-\[\]\<\>\?]", line.strip()):
                word = word.lower()
                if word:  # Ignore empty words
                    word_count[word] = word_count.get(word, 0) + 1
    return word_count


# --- Filter Out Words Below Frequency Threshold ---
def filter_low_frequency_words(word_count, min_word_count=5):
    """
    Filter out words that appear less than the specified minimum count.
    """
    return {word: count for word, count in word_count.items() if count >= min_word_count}


# --- Main Parallel Text Processing ---
def parallel_text_processing(_granularity, _broadcast_rate):
    """
    Main function to handle parallel word counting using MPI.
    """
    min_word_count = 5

    print("granularity =", _granularity)
    print("broadcast_rate =", _broadcast_rate)

    # Initialize MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Start measuring time
    start_time = time.time()

    # Prepare initial data
    dic_joint = {}
    update_dic = {}
    count = 0
    broadcast_count = 0

    # Root process (rank 0) handles file discovery and distribution
    if rank == 0:
        files = gather_files_to_process()
        files_per_process = distribute_files_across_processes(files, size)
    else:
        files_per_process = None

    # Scatter the file lists to processes
    files_for_this_process = comm.scatter(files_per_process, root=0)

    # Process files and count word frequencies
    for file_path in files_for_this_process:
        update_dic = process_file_for_word_counts(file_path)
        dic_joint = merge_word_counts(dic_joint, update_dic)
        count += 1

        # If granularity threshold reached, gather and merge results
        if count >= _granularity:
            dic_joint, broadcast_count = gather_and_merge_results(dic_joint, update_dic, comm, rank, size, _broadcast_rate, broadcast_count)
            count = 0
            update_dic = {}

    # Final merge for remaining results
    dic_joint, broadcast_count = gather_and_merge_results(dic_joint, update_dic, comm, rank, size, _broadcast_rate, broadcast_count)

    # Filter out low-frequency words on rank 0
    if rank == 0:
        dic_joint = filter_low_frequency_words(dic_joint, min_word_count)

    # Print the final dictionary size and processing time
    if rank == 0:
        print("Final word count dictionary length:", len(dic_joint))

    # End measuring time and return the duration
    end_time = time.time()
    execution_time = end_time - start_time
    return execution_time


# --- Gather and Merge Results ---
def gather_and_merge_results(dic_joint, update_dic, comm, rank, size, broadcast_rate, broadcast_count):
    """
    Gather word count dictionaries from all processes, merge them, and broadcast the result.
    """
    dic_list = comm.gather(update_dic, root=0)
    if rank == 0:
        for dic in dic_list:
            dic_joint = merge_word_counts(dic_joint, dic)

    if broadcast_count == 0:
        dic_joint = comm.bcast(dic_joint, root=0)
        if rank == 0:
            print("Broadcasting merged dictionary")

    broadcast_count = (broadcast_count + 1) % broadcast_rate
    return dic_joint, broadcast_count


# --- Gathering Files to Process (Rank 0) ---
def gather_files_to_process():
    """
    Gather all file paths to process from different directories.
    """
    path1 = "../../wiki_dataset_statistics/data/1of2"
    path2 = "../../wiki_dataset_statistics/data/2of2"
    path3 = "../../wiki_dataset_statistics/data/fullEnglish"

    files1 = [os.path.join(path1, f) for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f))]
    files2 = [os.path.join(path2, f) for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f))]
    files3 = list_files_walk(path3)

    return files1 + files2 + files3

if __name__ == "__main__":
    # Parse command-line arguments
    # args = parse_arguments()
    # gran = args.granularity
    # br = args.broadcast_rate

    # Run the combined optimization
    best_params = optimize_parameters()

    # Output the best combination of parameters found
    print(
        f"Best parameters found: Granularity = {best_params[0]}, Broadcast Rate = {best_params[1]}, Num Processes = {int(best_params[2])}")