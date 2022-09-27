import requests

REST_URL = "http://localhost:8090/tasks/create/file"
SAMPLE_FILE = "/home/b793170/filename.exe"
HEADERS = {"Authorization": "Bearer pxJLRqiTfxz0PNNhGLdoew"}

with open(SAMPLE_FILE, "rb") as sample:
    files = {"file": ("temp_file_name", sample)}
    r = requests.post(REST_URL, headers=HEADERS, files=files)


task_id = r.json()["task_id"]