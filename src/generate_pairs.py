""" generate pairs from specified dataset to evaluate"""

import os
import sys
import time
import random
import argparse
from itertools import combinations

def generate_all_pairs(data_dir, pairs_filepath, num_person, num_sample_per_person):

    start_time = time.time()
    #===================#
    # create pairs file #
    #===================#
    if not os.path.exists(pairs_filepath):
        with open(pairs_filepath,"a") as f:
            f.write(str(data_dir) + "\t" + str(num_person) + '\t' + str(num_sample_per_person) + "\n")

    all_img_path_list = []
    id_dir_path_list = [os.path.join(data_dir, path) for path in os.listdir(data_dir)]
    random.shuffle(id_dir_path_list)
    
    #==================================#
    # eumerate all path and get sample #
    #==================================#
    for i, id_dir_path in enumerate(id_dir_path_list):
        if num_person is not None and i >= num_person:
            break
        img_per_id_path_list = [os.path.join(id_dir_path, path) for path in os.listdir(id_dir_path)]
        random.shuffle(img_per_id_path_list)
        for j, img_path in enumerate(img_per_id_path_list):
            if num_sample_per_person is not None and j >= num_sample_per_person:
                break
            all_img_path_list.append(img_path)
    
    #======================#
    # get all combinations #
    #======================#
    combinations_list = list(combinations(all_img_path_list, 2))
    random.shuffle(combinations_list)
    
    #==================#
    # write pairs file #
    #==================#
    with open(pairs_filepath, "a") as f:
        for pair in combinations_list:
            if pair[0].split('/')[-2] == pair[1].split('/')[-2]:
                name = pair[0].split('/')[-2]
                path_1 = pair[0]
                path_2 = pair[1]
                f.write(name + "\t" + path_1 + "\t" + path_2 + "\n") 
            else:
                name_1 = pair[0].split('/')[-2]
                name_2 = pair[1].split('/')[-2]
                path_1 = pair[0]
                path_2 = pair[1]
                f.write(name_1 + "\t" + path_1 + "\t" + name_2 + '\t'+ path_2 + "\n") 
    run_time = time.time() - start_time
    return run_time, len(combinations_list)

def generate_pairs_with_balance(data_dir, pairs_filepath, repeat_times):
    """
        (1) generate match pairs
        (2) generate dismatch pairs
    """
    start_time = time.time()
    #===================#
    # create pairs file #
    #===================#
    if not os.path.exists(pairs_filepath):
        with open(pairs_filepath,"a") as f:
            f.write(str(data_dir) + "\t" + 'repeat times :\t' + str(repeat_times) + "\n")
    
    #======================#
    # generate match pairs #
    #======================#
    cnt = 0
    for _ in range(repeat_times):
        # print(cnt)
        for name in os.listdir(data_dir):
            name_path = os.path.join(data_dir, name)
            if not os.path.isdir(name_path):
                continue
            file_list = []
            for _file in os.listdir(name_path):
                file_list.append(os.path.join(name_path, _file))
            if len(file_list) < 1:
                continue
            try:
                with open(pairs_filepath, 'a') as f: 
                    path_1 = random.choice(file_list)
                    path_2 = random.choice(file_list)
                    f.write(name + '\t' + path_1 + '\t' + path_2 + '\n')
                    cnt += 1
            except Exception as e:
                # print(e)
                continue

        #==========================#
        # generate dis-match pairs #
        #==========================#
        for i, name in enumerate(os.listdir(data_dir)):
            # print(i)
            name_path = os.path.join(data_dir, name)
            if not os.path.isdir(name_path):
                continue

            remaining = os.listdir(data_dir)
            del remaining[i]
            other_dir = random.choice(remaining)

            name_path = os.path.join(data_dir, name)
            other_path = os.path.join(data_dir, other_dir)
            name_path_list = [os.path.join(name_path, path) for path in os.listdir(name_path)]
            other_path_list = [os.path.join(other_path, path) for path in os.listdir(other_path)]

            if len(name_path_list) == 0 or len(other_path_list) == 0:
                continue

            with open(pairs_filepath, 'a') as f:
                path_1 = random.choice(name_path_list)
                path_2 = random.choice(other_path_list)
                f.write(name + '\t' + path_1 + '\t' + other_dir + '\t' + path_2 + '\n')
                cnt += 1
    use_time = time.time() - start_time
    return use_time, cnt

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', type=str,
        help='Directory of evaluation dataset.')
    parser.add_argument('pairs_path', type=str,
        help='File path of the generated pairs.')
    parser.add_argument('--sample_type', type=int,
        help='Sample type of the task. 0:balance pos/neg, 1:sample by person and img per person', default=0)
    parser.add_argument('--repeat_times', type=int,
        help='Repeat times of generation, this argument only be used when --sample type is 0.', default=10)
    parser.add_argument('--num_person',type=int, 
        help='Number of person to sample, this argument only be used when --sample type is 1.', default=None)
    parser.add_argument('--num_sample', type=int,
        help='Number of sample per person, this argument only be used when --sample type is 1.', default=20)

    return parser.parse_args(argv)   

def main(args):
    print('** start generate pairs.')
    print('** use dataset : {}'.format(args.data_dir))
    print('** sample type : {}'.format(args.sample_type))
    if args.sample_type == 0:
        try:
            run_time, num_pairs = generate_pairs_with_balance(args.data_dir, args.pairs_path, args.repeat_times)
            print('** repeat times : {}'.format(args.repeat_times))
            print('** generating complete.')
            print('** thre are totally {} pairs.'.format(num_pairs))
            print('** save file to {}'.format(args.pairs_path))
            print('** use time : {}s'.format(run_time))
        except Exception as e:
            raise e
            print('*** Failed to generate ***')
            print(e)
    elif args.sample_type == 1:
        try:
            run_time, num_combinations = generate_all_pairs(args.data_dir, args.pairs_path, args.num_person, args.num_sample)
            print('** num_person : {}'.format(args.num_person))
            print('** num_sample : {}'.format(args.num_sample))
            print('** generating complete.')
            print('** thre are totally {} combinations.'.format(num_combinations))
            print('** save file to {}'.format(args.pairs_path))
            print('** use time : {}s'.format(run_time))
        except Exception as e:
            print('*** Failed to generate ***')
            print(e)
       
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))