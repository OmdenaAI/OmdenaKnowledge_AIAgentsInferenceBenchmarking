import argparse
import pandas as pd
import json
import random


def load_csv(file_path):
    """Load the crop yield dataset from a CSV file."""
    return pd.read_csv(file_path)


def create_jsonl(df, num_questions, output_file):
    """
    Generate a JSONL file with prompts and completions from the dataset.

    Args:
        df (pd.DataFrame): The dataset as a pandas DataFrame.
        num_questions (int): Number of questions to extract.
        output_file (str): The output JSONL filename.
    """
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Select the specified number of rows
    selected_rows = df.head(num_questions)

    # Open the output file
    with open(output_file, 'w') as jsonl_file:
        for _, row in selected_rows.iterrows():
            # Create the prompt
            prompt = (
                f"Given precipitation of {row['Precipitation (mm day-1)']} mm/day, "
                f"specific humidity of {row['Specific Humidity at 2 Meters (g/kg)']} g/kg, "
                f"relative humidity of {row['Relative Humidity at 2 Meters (%)']}%, "
                f"and temperature of {row['Temperature at 2 Meters (C)']}°C, "
                f"predict yield for crop {row['Crop']}."
            )

            # Create the completion
            completion = f"Yield is {row['Yield']}"

            # Write to JSONL
            jsonl_file.write(
                json.dumps({"prompt": prompt, "completion": completion}) + "\n"
            )


def main():
    # Set up argparse
    parser = argparse.ArgumentParser(description="Generate JSONL from crop yield dataset.")
    parser.add_argument('--crop_yield', required=True, help="Path to the crop yield CSV file.")
    parser.add_argument('--num_questions', type=int, required=True, help="Number of question/answer pairs to generate.")
    parser.add_argument('--output', required=True, help="Output JSONL file name.")

    # Parse arguments
    args = parser.parse_args()

    # Load the dataset
    df = load_csv(args.crop_yield)

    # Generate the JSONL file
    create_jsonl(df, args.num_questions, args.output)

    print(f"JSONL file with {args.num_questions} questions saved to {args.output}")


if __name__ == "__main__":
    main()
