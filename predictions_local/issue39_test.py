from neurosityprocessor import NeurosityDataProcessor
from neurosity import NeurositySDK
import time
from datetime import datetime
if __name__ == "__main__":
    headset = NeurosityDataProcessor()
    
    ts = 1783552809334 / 1000
    t = 1783552496871 / 1000  # milliseconds -> seconds
    print(datetime.fromtimestamp(ts))
    print(datetime.fromtimestamp(t))
    # time.sleep(10)

    try:
        output = headset.get_tensor()
        time.sleep(10)
        print(output)
    except Exception as e:
        print(f"Error: {e}")


