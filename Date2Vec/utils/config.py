import argparse
import os
from time import strftime
import shutil


# TODO: Function to create an output directory
def create_output_dir(args):
    # If output_dir is not provided
    if args.output_dir is None:
        # Save corresponding hyperparameters
        if args.save_log:  # Only enter this section if save_log is True

            args.output_dir = os.path.join('experiments')  # Set base directory for experiments
            # Find the largest exp number and increment it by 1
            current_exp = 0
            if os.path.exists(args.output_dir):
                exp_values = [int(f[3:]) for f in os.listdir(args.output_dir) if f.startswith('exp')]
                current_exp = max(exp_values) + 1 if exp_values else 0

            # If a specific experiment number is provided and is less than the current exp, set to exp_num
            if args.exp_num != -1 and args.exp_num < current_exp:
                current_exp = args.exp_num

            args.output_dir = os.path.join(args.output_dir, 'exp{}'.format(current_exp))
    else:
        # If output_dir is provided, check if it exists
        if not os.path.exists(args.output_dir):
            print(f'The output path {args.output_dir} does not exist.')
        else:
            shutil.rmtree(args.output_dir)  # Delete all files in the existing directory

    # Get current time for directory and file naming
    current_time = strftime('%Y-%m-%d_%H-%M-%S')

    # Create the output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create a time file to log the current time
    with open(os.path.join(args.output_dir, f'{current_time}.time'), 'a+'):
        pass

    # Create a README file and write the description into it
    with open(os.path.join(args.output_dir, 'README'), 'a+') as f:
        f.write(args.desc)

    return args
