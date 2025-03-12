import arff
import pandas as pd

def convert_arff_to_csv(input_arff, output_csv):
    # Load the ARFF file
    with open(input_arff, 'r') as f:
        dataset = arff.load(f)
        
    # Extract attribute names
    columns = [attr[0] for attr in dataset['attributes']]
    
    # Convert data to a Pandas DataFrame
    df = pd.DataFrame(dataset['data'], columns=columns)
    
    # Write DataFrame to CSV
    df.to_csv(output_csv, index=False)
    print(f"Conversion complete! CSV saved to: {output_csv}")

if __name__ == "__main__":
    # Update 'bmt_dataset.arff' and 'bmt_dataset.csv' to your actual filenames
    input_arff = "data/raw/bmt_dataset.arff"
    output_csv = "data/processed/bmt_dataset.csv"

    convert_arff_to_csv(input_arff, output_csv)