import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|test|dataset|help]")
        sys.exit(1)
    
    if sys.argv[1] == "train":
        print("Training the model...")
    elif sys.argv[1] == "test":
        print("Testing the model...")
    elif sys.argv[1] == "dataset":
        from data.generate import generate_data
        generate_data("data/train.csv")
    else:
        print("Usage: python main.py [train|test|dataset|help]")
        sys.exit(1)

if __name__ == "__main__":
    main()