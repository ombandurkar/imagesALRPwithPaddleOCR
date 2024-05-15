# #comparing the original result vs generated ocr result with Levenshetin algorithm and dynamic programming
# def calculate_accuracy(original_plate, ocr_plate):
#     """
#     Calculate the accuracy between original and OCR plates.
#     """
#     original_plate = original_plate.lower()  # Convert to lowercase for case-insensitive comparison
#     ocr_plate = ocr_plate.lower()

#     # Calculate Levenshtein distance
#     if len(original_plate) == 0:
#         return 0.0 if len(ocr_plate) == 0 else 0.0
#     if len(ocr_plate) == 0:
#         return 0.0 if len(original_plate) == 0 else 0.0

#     matrix = [[0] * (len(ocr_plate) + 1) for _ in range(len(original_plate) + 1)]
#     for i in range(len(original_plate) + 1):
#         matrix[i][0] = i
#     for j in range(len(ocr_plate) + 1):
#         matrix[0][j] = j

#     for i in range(1, len(original_plate) + 1):
#         for j in range(1, len(ocr_plate) + 1):
#             cost = 0 if original_plate[i - 1] == ocr_plate[j - 1] else 1
#             matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + cost)

#     return 1 - matrix[-1][-1] / max(len(original_plate), len(ocr_plate))

# def compare_results(original_file, ocr_file, output_file):
#     """
#     Compare original OCR results with OCR results and write accuracy to output file.
#     """
#     with open(original_file, 'r') as original:
#         original_results = original.readlines()
#     with open(ocr_file, 'r') as ocr:
#         ocr_results = ocr.readlines()

#     with open(output_file, 'w') as output:
#         output.write("Original Plate\tOCR Plate\tAccuracy\n")
#         for original_plate, ocr_plate in zip(original_results, ocr_results):
#             original_plate = original_plate.strip()
#             ocr_plate = ocr_plate.strip()
#             accuracy = calculate_accuracy(original_plate, ocr_plate)
#             output.write(f"{original_plate}\t{ocr_plate}\t{accuracy:.2f}\n")

#     print("Comparison results have been written to", output_file)

# # Usage
# compare_results('original_results.txt', 'main_ocr_results.txt', 'comparison_results.txt')


# ********************** comparison result with avg accuracy included *************************


def calculate_accuracy(original_plate, ocr_plate):
    """
    Calculate the accuracy between original and OCR plates.
    """
    original_plate = original_plate.lower()  # Convert to lowercase for case-insensitive comparison
    ocr_plate = ocr_plate.lower()

    # Calculate Levenshtein distance
    if len(original_plate) == 0:
        return 0.0 if len(ocr_plate) == 0 else 0.0
    if len(ocr_plate) == 0:
        return 0.0 if len(original_plate) == 0 else 0.0

    matrix = [[0] * (len(ocr_plate) + 1) for _ in range(len(original_plate) + 1)]
    for i in range(len(original_plate) + 1):
        matrix[i][0] = i
    for j in range(len(ocr_plate) + 1):
        matrix[0][j] = j

    for i in range(1, len(original_plate) + 1):
        for j in range(1, len(ocr_plate) + 1):
            cost = 0 if original_plate[i - 1] == ocr_plate[j - 1] else 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + cost)

    return 1 - matrix[-1][-1] / max(len(original_plate), len(ocr_plate))

def compare_results(original_file, ocr_file, output_file):
    """
    Compare original OCR results with OCR results and write accuracy to output file.
    """
    total_accuracy = 0.0
    total_plates = 0

    with open(original_file, 'r') as original:
        original_results = original.readlines()
    with open(ocr_file, 'r') as ocr:
        ocr_results = ocr.readlines()

    with open(output_file, 'w') as output:
        output.write("Original Plate\tOCR Plate\tAccuracy\n")
        for original_plate, ocr_plate in zip(original_results, ocr_results):
            original_plate = original_plate.strip()
            ocr_plate = ocr_plate.strip()
            accuracy = calculate_accuracy(original_plate, ocr_plate)
            total_accuracy += accuracy
            total_plates += 1
            output.write(f"{original_plate}\t{ocr_plate}\t{accuracy:.2f}\n")

        avg_accuracy = total_accuracy / total_plates if total_plates > 0 else 0.0
        output.write(f"\nAverage Accuracy: {avg_accuracy:.2f}")

    print("Comparison results have been written to", output_file)

# Usage
compare_results('original_results.txt', 'main_ocr_results.txt', 'comparison_results.txt')