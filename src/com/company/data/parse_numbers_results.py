from posixpath import split

matches = ["Prerequisites", "Dataset Size", "Number of missing values", "Column Predictors", "Statistics of predicted value"]
file1 = open('CCPP_results_comparison.txt', 'r')
file2 = open('YellowStoneElev_results_comparison.txt', 'r')
file3 = open('combined_multiple_results_comparison.txt', 'r')

Lines = file1.readlines()

for line in Lines:
    if not any(x in line for x in matches):
    # if(not line.__contains__("Prerequisites")) :
        if( "Predicted:" in line):
            print(line)
        elif(len(line.split(":")) == 2 and len(line.split(":")[1]) == 1):
            if("\t" not in line):
                print()
            print (line, end="")
        elif(len(line.split(":")) == 2):
            x = line.count("\t")
            for i in range(0, x):
                print("\t", end="")
            print( line.split(":")[1].replace(".", ",").replace("%", "").strip())
    