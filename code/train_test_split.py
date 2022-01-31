#train_test_split

import pandas 
import argparse
import os



def main(x_path, y_path):
    print(x_path)
    print(y_path)

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description='List the content of a folder')

    # Add the arguments
    my_parser.add_argument('X_path',
        default = os.path.join(r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\data\modelling", 'X_77_qlesq_sr__final_extval'),
        help='the path to X')

    # Add the arguments
    my_parser.add_argument('Y_path',
        default = os.path.join(r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\data\modelling", 'y_77_qlesq_sr__final__targets'),
        help='the path to y')

    args = my_parser.parse_args()
    main(args.X_path, args.Y_path)