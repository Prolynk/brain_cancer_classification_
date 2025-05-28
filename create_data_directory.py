import os
import shutil

data_dir = "dataset"

for folder in os.listdir(data_dir):
    if ' ' in folder:
        # Get only the main class name, discard MRI type like T1, T2, etc.
        main_class = folder.split(' ')[0]
        mri_type = folder.split(' ')[1]

        src_folder_path = os.path.join(data_dir, folder)
        dst_folder_path = os.path.join(data_dir, main_class)

        # Create destination folder if it doesn't exist
        os.makedirs(dst_folder_path, exist_ok=True)

        for file in os.listdir(src_folder_path):
            src_file_path = os.path.join(src_folder_path, file)
            base_name, ext = os.path.splitext(file)

            # Create new filename with MRI type tag
            new_file_name = f"{base_name}_{mri_type}{ext}"
            dst_file_path = os.path.join(dst_folder_path, new_file_name)

            # Avoid overwriting
            if os.path.exists(dst_file_path):
                print(f"File already exists, skipping: {dst_file_path}")
                continue

            try:
                shutil.move(src_file_path, dst_file_path)
            except Exception as e:
                print(f"Error moving file {src_file_path}: {e}")

        # Try to remove the now-empty source folder
        try:
            os.rmdir(src_folder_path)
        except OSError as e:
            print(f"Could not remove folder {src_folder_path}: {e}")
