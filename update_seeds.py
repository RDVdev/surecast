import glob

files = glob.glob("phase*.py") + ["robustness_audit.py"]
for f in files:
    with open(f, 'r') as file:
        content = file.read()
    
    if "set_seed(42)" not in content and "set_seed" in content:
        if 'if __name__ == "__main__":' in content:
            content = content.replace('if __name__ == "__main__":', 'if __name__ == "__main__":\n    set_seed(42)')
            with open(f, 'w') as file:
                file.write(content)
            print(f"Updated {f}")
