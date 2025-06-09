import os

folder = os.path.join(os.getcwd(), "tox21", "models")
print(folder)
for filename in os.listdir(folder):
    if "wd0seed" in filename:
        fixed_filename = filename.replace("wd0seed", "wd0_bias_seed")
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, fixed_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {fixed_filename}")
    elif "biasseed" in filename:
        fixed_filename = filename.replace("biasseed", "bias_seed")
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, fixed_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {fixed_filename}")
    elif r"[1024]" in filename:
        fixed_filename = filename.replace(r"[1024]", r"1024")
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, fixed_filename)
        try:
            os.rename(old_path, new_path)
        except:
            print("Tried to Rename:", filename)
            print("FAILED -> file already exists:", fixed_filename)
            continue
        print(f"Renamed: {filename} → {fixed_filename}")

folder = os.path.join(os.getcwd(), "sider", "models")
print(folder)
for filename in os.listdir(folder):
    if "wd0seed" in filename:
        fixed_filename = filename.replace("wd0seed", "wd0_bias_seed")
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, fixed_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {fixed_filename}")
    elif "biasseed" in filename:
        fixed_filename = filename.replace("biasseed", "bias_seed")
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, fixed_filename)
        try:
            os.rename(old_path, new_path)
        except:
            print("Tried to Rename:", filename)
            print("FAILED -> file already exists:", fixed_filename)
            continue
        
        
        print(f"Renamed: {filename} → {fixed_filename}")

