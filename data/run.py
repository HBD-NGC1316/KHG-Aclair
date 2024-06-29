##########################################################################################
## Uses argparse to select a dataset and sequentially executes scripts associated with the chosen dataset.
## Executes "step5" scripts with user-specified relational cardinality ranges as arguments, if applicable.
##  Displays an error if an invalid dataset is provided.
##########################################################################################



import argparse
import sys
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create HKG")
    parser.add_argument('--dataset', type=str, default='movielens',
                        help="available datasets: [movielens, last-fm, amazon-book]")
    parser.add_argument('--rel_min', type=str, default='0',
                        help="input the reation cardinality min range. e.g.) '0' ")
    parser.add_argument('--rel_max', type=str, default='10000',
                        help="input the reation cardinality max range. e.g.) '0' ")

    args = parser.parse_args()

    dataset = args.dataset
    rel_min = args.rel_min
    rel_max = args.rel_max

    # Current working die
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scripts = ['step1_common_item.py', 'step2_item_group.py', 'step3_hkg_creater.py', 'step4_encoder.py', 'step5_rel_cardi.py']


    if dataset == 'movielens':
        for script in scripts:
            print("script - ", script)
            script_path = os.path.join(current_dir, 'movielens', script)
            if "step5" in script:
                os.system(f"python3 {script_path} --rel_min {rel_min} --rel_max {rel_max}")
            else:
                os.system(f"python3 {script_path}")
    
    elif dataset == 'last-fm':
        for script in scripts:
            print("script - ", script)
            script_path = os.path.join(current_dir, 'last-fm', script)
            if "step5" in script:
                os.system(f"python3 {script_path} --rel_min {rel_min} --rel_max {rel_max}")
            else:
                os.system(f"python3 {script_path}")
        
    elif dataset == 'amazon-book':
        for script in scripts:
            print("script - ", script)
            script_path = os.path.join(current_dir, 'amazon-book', script)
            if "step5" in script:
                os.system(f"python3 {script_path} --rel_min {rel_min} --rel_max {rel_max}")
            else:
                os.system(f"python3 {script_path}")
        
    else:
        sys.exit("Error: Invalid dataset provided. Please provide one of the available datasets: [movielens, last-fm, amazon-book]")
