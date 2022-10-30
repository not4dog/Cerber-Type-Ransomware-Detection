from pathlib import Path

# 참고 : https://brownbears.tistory.com/415 & https://engineer-mole.tistory.com/191


def get_project_root_directory():
    # os.path는 경로를 문자열로 다루지만, Pathlib은 경로를 객체로 다룬다
    # __file__ : 현재 파일, Path : 현재 파일 경로 객체화, resolve : os.path.abspath처럼 절대경로 반환

    # 현재 경로 ( ex) C:\Users\hyunf\PycharmProjects\Cerber-Type-Ransomware-Detection\modeling )
    data_source_directory = Path(__file__).resolve()

    # 실행중인 스크립트의 부모 디렉토리 경로 -> C:\Users\hyunf\PycharmProjects\Cerber-Type-Ransomware-Detection
    project_root_directory = data_source_directory.parent.parent

    return project_root_directory

def check_file_exist(file_path):
    # 경로에 파일 있는가 있으면 
    try:
        file = open(file_path)
        return True

    except FileNotFoundError:
        return False
