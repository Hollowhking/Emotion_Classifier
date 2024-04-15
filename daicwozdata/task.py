import csv

def count_zeros_in_second_column(csv_path):
    # Initialize counter for zeros
    zero_count = 0
    one_count = 0
    between59 = 0
    higher15 = 0
    count = 0
    # Read data from the CSV file
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Skip header row if it exists
        next(csv_reader, None)
        
        # Count zeros in the second column
        for row in csv_reader:
            if len(row) > 1 and row[1] == '0':
                zero_count += 1
            elif len(row) > 1 and row[1] == '1':
                one_count += 1
            if (len(row) > 1) and int(row[2]) > 5 and int(row[2]) < 9:
                between59 += 1
    return zero_count,one_count,between59

# Example usage
csv_path = 'combined_sorted_outputfull.csv'  # Replace with the actual file path
zeros_count,ones_count,between5_9 = count_zeros_in_second_column(csv_path)

print(f"Number of times Normal people appear in the dataset: {zeros_count}")
print(f"Number of times depressed people appear in the dataset: {ones_count}")
print(f"Number of times almost depressed (between 5-9) people appear in the dataset: {between5_9}")

