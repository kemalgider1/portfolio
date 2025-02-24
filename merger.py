import os


def main():
    current_dir = os.getcwd()
    files = [f for f in os.listdir(current_dir) if f.endswith('.py') and f != 'merged.py']
    print("Available Python scripts:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")

    selected = input("Enter the numbers of the scripts to merge (comma separated): ")
    selected_indexes = [int(x.strip()) - 1 for x in selected.split(",") if x.strip().isdigit()]
    selected_files = [files[i] for i in selected_indexes if 0 <= i < len(files)]

    merged_content = ""
    divider = "\n=======================================================\n"

    for filename in selected_files:
        with open(filename, 'r') as f:
            merged_content += f.read() + divider

    if merged_content.endswith(divider):
        merged_content = merged_content[:-len(divider)]

    output_filename = "merged.py"
    with open(output_filename, 'w') as f:
        f.write(merged_content)

    print(f"Merged file saved as {output_filename}")


if __name__ == "__main__":
    main()