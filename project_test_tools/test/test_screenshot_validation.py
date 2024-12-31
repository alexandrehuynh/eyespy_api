import sys
import os

# Add the parent directory of `project_test_tools` to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from project_test_tools.utils import validate_screenshots

if __name__ == "__main__":
    screenshots_folder = input("Enter the path to the screenshots folder: ")
    metadata_file = input("Enter the path to the JSON metadata file: ")

    # Run validation
    results = validate_screenshots(screenshots_folder, metadata_file)

    # Print summary
    print("\nValidation Results:")
    print(f"Aligned screenshots: {len(results['aligned'])}")
    print(f"Mismatched timestamps: {len(results['mismatched'])}")
    print(f"Missing metadata: {len(results['missing_metadata'])}\n")

    # Detailed mismatches
    if results["mismatched"]:
        print("Mismatched Timestamps:")
        for mismatch in results["mismatched"]:
            print(mismatch)

    # Missing metadata
    if results["missing_metadata"]:
        print("\nScreenshots with Missing Metadata:")
        for missing in results["missing_metadata"]:
            print(missing)