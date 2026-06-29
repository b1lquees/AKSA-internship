"""
Task: File Reader
- read a .txt file 
- count the number of lines, words and characters. """
import sys 
def main():
    count = 0
    total_lines = 0
    word_count = 0
    char_count = 0
    # get the file name 
    file_name = input("Enter the file name: ")
    if not file_name.endswith(".txt"):
        sys.exit("Not a txt file")
    try:
        # use with open so you dont have to close it explicitly
        with open(file_name, "r") as file: 
            for line in file: # iterate through each line in the file 
                total_lines +=1
                stripped = line.strip() # strip the whitespace and inc the var count by 1 
                if stripped: # this is to skip lines with spaces 
                    count += 1
                # count the words and chars in each line 
                word_count += len(line.split())
                char_count += len(line.strip()) # to exclude white spaces
            print(f"Number of lines in the file: {total_lines}")
            print(f"Number of non-empty lines in the file: {count}")
            print(f"Number of words in the file: {word_count}")
            print(f"Number of characters in the file: {char_count}")
            
    except FileNotFoundError:
        sys.exit("File does not exist")

if __name__ == "__main__":
    main()
