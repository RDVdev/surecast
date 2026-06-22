import os
import sys

def check_data_dependency(path, requirement_msg):
    if not os.path.exists(path):
        print(f"\n[ERROR] NOT YET COMPUTED — requires {requirement_msg} ({path})")
        sys.exit(1)
