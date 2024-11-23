import os
import re
import time
from mpi4py import MPI
import argparse


# --- Parsing Command-Line Arguments ---
def parse_arguments():
    parser = argparse.ArgumentParser(description='Parallel texts processing app')
    parser.add_argument('--gran', action="store", dest='granularity', default=1)
    parser.add_argument('--bcast_rate', action="store", dest='broadcast_rate', default=1)
    return parser.parse_args()


# --- List All Files in Directory and Subdirectories ---
def list_files_walk(start_path='.'):
    files = []
    for root, dirs, files_in_dir in os.walk(start_path):
        for file in files_in_dir:
            files.append(os.path.join(root, file))
    return files


# --- Divide File List Between Processes ---
def divide_files_among_processes(filenames, processes_count):
    divided_files = []
    for i in range(processes_count):
        files_for_process = []
        curr_filename_index = i
        while curr_filename_index < len(filenames):
            files_for_process.append(filenames[curr_filename_index])
            curr_filename_index += processes_count
        divided_files.append(files_for_process)
    return divided_files


# --- Merge Two Dictionaries ---
def merge_dictionaries(d1, d2):
    for key in d2.keys():
        d1[key] = d1.get(key, 0) + d2[key]
    return d1


# --- Count Word Frequency in File ---
def process_file(f, dic_joint, update_dic):
    with open(f, 'r') as file:
        lines = file.readlines()
        for line in lines:
            for word in re.split(r"[ ,\.\(\)\;\|\"\d\\\/\+\-\[\]\<\>\?]", line.strip()):
                word = word.lower()
                if word:
                    update_dic[word] = update_dic.get(word, 0) + 1
    dic_joint = merge_dictionaries(dic_joint, update_dic)
    return dic_joint, update_dic


# --- Filter Words Based on Frequency ---
def filter_words(dic_joint, min_word_count=5):
    return {word: count for word, count in dic_joint.items() if count >= min_word_count}


# --- Main Parallel Text Processing Function ---
def parallel_text_processing():
    # Parse arguments
    args = parse_arguments()

    min_word_count = 5
    granularity = args.granularity
    broadcast_rate = args.broadcast_rate

    print("granularity =", granularity)
    print("broadcast_rate =", broadcast_rate)

    # Initialize MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Initialize dictionary
    dic_joint = {}
    update_dic = {}
    count = 0
    broadcast_count = 0

    if rank == 0:
        path1 = "../../wiki_dataset_statistics/data/1of2"
        path2 = "../../wiki_dataset_statistics/data/2of2"
        path3 = "../../wiki_dataset_statistics/data/fullEnglish"
        files1 = [os.path.join(path1, f) for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f))]
        files2 = [os.path.join(path2, f) for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f))]
        files3 = list_files_walk(path3)
        files = files1 + files2 + files3

        # Divide files across processes
        files_per_process = divide_files_among_processes(files, size)
    else:
        files_per_process = None

    # Scatter files to each process
    files_for_this_process = comm.scatter(files_per_process, root=0)

    if rank == 0:
        start_time = time.time()
        print("Data load took: ", time.time() - start_time)

    # Processing files
    for f in files_for_this_process:
        dic_joint, update_dic = process_file(f, dic_joint, update_dic)
        count += 1

        # Perform merge and broadcast based on granularity
        if count > granularity:
            if rank == 0:
                start_merge_time = time.time()
                print(f"Rank {rank} - Update: {len(update_dic)} Local: {len(dic_joint)}")

            dic_list = comm.gather(update_dic, root=0)
            update_dic = {}
            count = 0

            if rank == 0:
                for dic in dic_list:
                    dic_joint = merge_dictionaries(dic_joint, dic)

            if broadcast_count == 0:
                dic_joint = comm.bcast(dic_joint, root=0)
                if rank == 0:
                    print("Broadcasting dic_joint")

            broadcast_count = (broadcast_count + 1) % broadcast_rate

            if rank == 0:
                print(f"Merging dictionaries took: {time.time() - start_merge_time} seconds")

    # Final merging after processing all files
    if update_dic:
        dic_list = comm.gather(update_dic, root=0)
        if rank == 0:
            for dic in dic_list:
                dic_joint = merge_dictionaries(dic_joint, dic)

    # Final cleanup
    if rank == 0:
        dic_joint = filter_words(dic_joint, min_word_count)

        print("Final dictionary length:", len(dic_joint))

        # Print processing time
        print(f"Processing time for rank {rank}: {time.time() - start_time} seconds")


if __name__ == "__main__":
    parallel_text_processing()
