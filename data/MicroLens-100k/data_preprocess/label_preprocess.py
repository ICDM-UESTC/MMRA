import math

def log2_of_label(label):
    if label <= 0:
        raise ValueError("the label must be a positive integer")
    return math.log2(label)

if __name__ == "__main__":
    label = 8
    result = log2_of_label(label)
    print(f"log2({label}) = {result}")