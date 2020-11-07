import csv
csv_files = ['train.csv', 'eval.csv']
txt_files = ['train.txt', 'eval.txt']
for i in range(len(csv_files)):
    csv_file = csv_files[i]
    txt_file = txt_files[i]
    with open(txt_file, "w") as my_output_file:
        with open(csv_file, "r") as my_input_file:
            [my_output_file.write("\t".join(row)+'\n') for row in csv.reader(my_input_file)]
        my_output_file.close()
