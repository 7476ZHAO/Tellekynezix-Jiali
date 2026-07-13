import os
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from neurosity import NeurositySDK

class NeurosityDataProcessor:
    def __init__(self, email=None, password=None, device_id=None, env_path=None, status_callback=None):
        self.env_path = Path(env_path) if env_path else Path(__file__).resolve().parent.parent / ".env.txt"
        self.email = email
        self.password = password
        self.device_id = device_id
        self._client = None
        self._buffer = []
        self._buffer_size = 512
        self.device_state = "unknown"
        self.status_callback = status_callback

        if self.email is None or self.password is None or self.device_id is None:
            self._load_env()

    def _load_env(self):
        if self.env_path.exists():
            load_dotenv(self.env_path)

        self.email = self.email or os.getenv("NEUROSITY_EMAIL")
        self.password = self.password or os.getenv("NEUROSITY_PASSWORD")
        self.device_id = self.device_id or os.getenv("NEUROSITY_DEVICE_ID")

        missing = [name for name, value in (
            ("NEUROSITY_EMAIL", self.email),
            ("NEUROSITY_PASSWORD", self.password),
            ("NEUROSITY_DEVICE_ID", self.device_id),
        ) if not value]
        if missing:
            raise RuntimeError("Missing required Neurosity credentials: " + ", ".join(missing))

    def _create_client(self):
        if self._client is not None:
            return

        self._load_env()
        self._client = NeurositySDK({"device_id": self.device_id})
        self._client.login({
            "email": self.email,
            "password": self.password,
        })
        self._client.status(self._status_callback)

    def _status_callback(self, status):
        print("callback:", status)
        # self.device_state = status.get("state", "unknown")
        self.device_state = self.get_device_state_once()
        if self.status_callback:
            self.status_callback(self.device_state)
    
    def get_device_state_once(self):
        self._create_client()
        return self._client.status_once().get("state", "unknown")
    

    def _brainwaves_callback(self, data):
        if isinstance(data, dict) and "data" in data:
            samples = np.array(data["data"])
            if samples.ndim == 2:
                self._buffer.extend(samples.T.tolist())

    def get_tensor(self, buffer_size=None, timeout=15):
        buffer_size = buffer_size or self._buffer_size
        self._create_client()
        self._buffer = []

        unsubscribe = self._client.brainwaves_raw(self._brainwaves_callback)
        start = time.time()
        print("cached:", self.device_state)
        print("once:", self._client.status_once()["state"])
        while len(self._buffer) < buffer_size and time.time() - start < timeout:
            #check the state of device, if offline, stop getting data
            

            if self._client.status_once()["state"] != "online":
                unsubscribe()
                raise RuntimeError("Neurosity device went offline during acquisition.")

            time.sleep(0.1)

        
        if unsubscribe:
            unsubscribe()

        if not self._buffer:
            raise RuntimeError("Unable to capture Neurosity data within timeout period")

        data = np.array(self._buffer[-buffer_size:])
        if data.ndim == 1:
            data = data.reshape(buffer_size, 1)

        return torch.tensor(data, dtype=torch.float32)



# if __name__ == "__main__":
#     processor = NeurosityDataProcessor()

#     # Create the client and log in
#     processor._create_client()

#     # Test device status
#     print("Checking device status...")
#     time.sleep(2)
#     print("Device online:", processor.get_device_state_once())
#     print('test')
#     print(processor._client.get_from_path('status'))
#     status = processor._client.status_once()
    
#     print(status)
#     print(processor.get_tensor())
#     print(processor._client.get_info())
#     status = processor._client.status_once()
#     print(status["lastHeartbeat"])
#     status = processor._client.status_once()
#     print(status["lastHeartbeat"])


    # Uncomment to test EEG acquisition
    # print("Getting EEG tensor...")
    # tensor = processor.get_tensor()
    # print(tensor.shape)
    # print(tensor)
