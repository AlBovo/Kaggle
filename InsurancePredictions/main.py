import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|test|dataset|help]")
        sys.exit(1)
    
    if sys.argv[1] == "train":
        from models.model import train_model
        train_model()
        print("Training the model completed successfully.")
    elif sys.argv[1] == "test":
        from models.model import test_model
        test_model()
        print("Testing the model completed successfully.")
    elif sys.argv[1] == "dataset":
        from data.generate import generate_data, generate_test
        if len(sys.argv) < 3:
            print("Usage: python main.py dataset [data|test]")
            sys.exit(1)
        elif sys.argv[2] == "data":
            generate_data("data/train.csv")
        elif sys.argv[2] == "test":
            generate_test("data/test.csv")
    else:
        print("Usage: python main.py [train|test|dataset|help]")
        sys.exit(1)

if __name__ == "__main__":
    main()