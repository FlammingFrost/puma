import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from quantization import model_4bit, model_fp16, model_8bit

def main():
    # model_fp16.eval()
    # model_fp16.test()
    
    model_8bit.eval()
    model_8bit.test()
    
    model_4bit.eval()
    model_4bit.test()
    
    


if __name__ == "__main__":
    main()
