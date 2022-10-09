from pathlib import Path

def get_project_root_directory():
    # 파일 경로 확인 후 경로 복사하여 project_root_directory 변수로 대입

    data_source_directory = Path(__file__)
    project_root_directory = data_source_directory.parent.parent.parent
    return project_root_directory

def check_file_exist(file_path):
    # 경로에 파일 있는가
    try:
        file = open(file_path)
        return True

    except FileNotFoundError:
        return False