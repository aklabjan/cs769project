import sys
import csv
from statistics import mean 
import os

def get_metrics(raw_output):
    with open(raw_output + '.out') as f:
        lines = f.readlines()
    i = 0
    average = {}
    while i < len(lines):
        # if lines[i].startswith('Epoch:'):
        #     file.write(lines[i])
        #     i += 1
        if 'Eval results on test dataset' in lines[i]:
            for count in range(1,12):
                output = lines[i+count].split()
                if output[-3] not in average:
                    average[output[-3]] = []
                average[output[-3]].append(float(output[-1]))
            i += 12
        else:
            i += 1

    for metric in average:
        average[metric] = max(average[metric])

    out_file = os.path.join('clean_results',raw_output + '_clean.csv')
    with open(out_file, 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the header (field names)
        csv_writer.writerow(['Metric', 'Value'])

        # Write keys and values
        for key, value in average.items():
            csv_writer.writerow([key, value])

def get_data(raw_output):
    with open(raw_output + '.out') as f:
        lines = f.readlines()
    with open(raw_output + '_clean.txt', 'w') as file:
        i = 0
        while i < len(lines):
            if lines[i].startswith('Epoch:'):
                file.write(lines[i])
                i += 1
            if 'Eval results on dev dataset' in lines[i]:
                file.write(lines[i])
                for count in range(1,12):
                    file.write(lines[i+count])
                i += 12
            else:
                i += 1

if __name__ == "__main__":
    raw_output = sys.argv[1]
    get_metrics(raw_output)
    #get_data(raw_output)