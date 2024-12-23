import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|test|help]")
        sys.exit(1)
    
    if sys.argv[1] == "train":
        from models.model import train_model
        train_model()
        print("Training the model completed successfully.")
    elif sys.argv[1] == "test":
        from models.model import test_model
        test_model()
        print("Testing the model completed successfully.")
    else:
        print("Usage: python main.py [train|test|help]")
        sys.exit(1)

if __name__ == "__main__":
    main()