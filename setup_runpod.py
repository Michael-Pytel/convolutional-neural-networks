import os
import shutil
import tarfile
import time

def optimize_runpod_storage():
    source_archive = "/workspace/cinic-10.tar.gz"
    fast_dest = "/tmp/cinic10"
    
    if os.path.exists(fast_dest):
        print(f"Dataset already exists at {fast_dest}. Ready to train.")
        return

    print("Extracting dataset to fast local NVMe...")
    start = time.time()
    
    os.makedirs(fast_dest, exist_ok=True)
    
    with tarfile.open(source_archive, "r:gz") as tar:
        tar.extractall(path=fast_dest)
        
    print(f"Extraction complete in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    optimize_runpod_storage()