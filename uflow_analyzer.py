import sys, getopt
import re

import numpy as np
import matplotlib.pyplot as plt

def print_help():
    print ('python3 -m uflow_analyzer.py -i <uflow log to be parsed> -t <experiment title> -e <epochs>')

def single_plot(input_array, ylabel, xlabel, grid, title, filename):
    plt.plot(input_array, color="#409DAB", linestyle='-', linewidth=3.0)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(grid)
    plt.title(title)
    plt.savefig(filename, dpi=500)
    plt.cla()
    plt.clf()
    
def main(argv):
    log = None
    title = None
    epochs = None

    opts, args = getopt.getopt(argv, "hi:t:e:", ["input="])
    for opt, arg in opts:
        if opt == '-h':
            print_help()
            sys.exit()
        elif opt in ("-i", "--input"):
            log = arg
            print("Parsing log " + log)
        elif opt in ("-t", "--title"):
            title = arg
            print("Experiment title " + title)
        elif opt in ("-e", "--epochs"):
            epochs = arg
            print("Epochs " + str(epochs))

    if log is None or title is None or epochs is None:
        print_help()
        sys.exit()

    pattern = re.compile("^[0-9]+\s--\stotal-loss:\s[0-9]+.[0-9]+,\scensus-loss:\s[0-9]+.[0-9]+,\sdata-time:\s[0-9]+.[0-9]+,\slearning-rate:\s[0-9]+.[0-9]+,\s(selfsup-loss:\s[0-9]+.[0-9]+,\s)*smooth1-loss:\s[0-9]+.[0-9]+,\strain-time:\s[0-9]+.[0-9]+")

    # Array of loss per epoch
    total_loss = np.empty(int(epochs)+1, dtype=float)
    census_loss = np.empty(int(epochs)+1, dtype=float)
    smooth1_loss = np.empty(int(epochs)+1, dtype=float)
    selfsup_loss = np.empty(int(epochs)+1, dtype=float)
    last_epoch = 0

    for i, line in enumerate(open(log)):
        for match in re.finditer(pattern, line):
            array_of_line = match.group().split()
            total_loss_str = str(array_of_line[3]).replace(",", "")
            census_loss_str = str(array_of_line[5]).replace(",", "")
            data_time_str = str(array_of_line[7]).replace(",", "")
            learning_rate_str = str(array_of_line[9]).replace(",", "")

            if len(array_of_line) < 16:
                selfsup_loss_str = "0"
                smooth1_loss_str = str(array_of_line[11]).replace(",", "")
                train_time_str = str(array_of_line[13])
            else:
                selfsup_loss_str = str(array_of_line[11]).replace(",", "")
                smooth1_loss_str = str(array_of_line[13]).replace(",", "")
                train_time_str = str(array_of_line[15])
                
            # print("Epoch " + str(last_epoch) + "/" + str(array_of_line[0]) + " total-loss " + total_loss_str + " census-loss " + census_loss_str + " data-time " + data_time_str + " learning-rate " +
            #       learning_rate_str + " selfsup-loss " + selfsup_loss_str + " smooth1-loss " + smooth1_loss_str + " train-time " + train_time_str)
            total_loss[last_epoch] = float(total_loss_str)
            census_loss[last_epoch] = float(census_loss_str)
            smooth1_loss[last_epoch] = float(smooth1_loss_str)
            selfsup_loss[last_epoch] = float(selfsup_loss_str)
            last_epoch += 1
            
            

    single_plot(total_loss[:last_epoch + 1], 'Loss', 'Epoch', True, str(title) + ' - Total Loss per Epoch', 'total_loss.png')
    single_plot(census_loss[:last_epoch + 1], 'Loss', 'Epoch', True, str(title) + ' - Census Loss per Epoch', 'census_loss.png')
    single_plot(smooth1_loss[:last_epoch + 1], 'Loss', 'Epoch', True, str(title) + ' - Smooth Loss per Epoch', 'smooth_loss.png')
    single_plot(selfsup_loss[:last_epoch + 1], 'Loss', 'Epoch', True, str(title) + ' - Self Supervision Loss per Epoch', 'selfsup_loss.png')

if __name__ == "__main__":
    main(sys.argv[1:])
