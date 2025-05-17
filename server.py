import os
import uuid
import json
import threading
from flask import Flask, request, jsonify, send_file
from scroing_csv import ScoringCSV  # 假設你將 ScoringCSV 抽成獨立模組

app = Flask(__name__)

# 任務儲存根目錄
TASK_ROOT = './Tasks'
os.makedirs(TASK_ROOT, exist_ok=True)

def write_status(path, status, message=None):
    with open(path, 'w') as f:
        json.dump({'status': status, 'message': message}, f)

def read_status(path):
    if not os.path.exists(path):
        return {'status': 'not_found'}
    with open(path) as f:
        return json.load(f)

# 任務上傳與啟動
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # 建立唯一任務 ID 與資料夾
    task_id = str(uuid.uuid4())
    task_dir = os.path.join(TASK_ROOT, task_id)
    os.makedirs(task_dir, exist_ok=True)

    # 檔案與狀態路徑
    input_path = os.path.join(task_dir, 'input.csv')
    output_path = os.path.join(task_dir, 'output.csv')
    status_path = os.path.join(task_dir, 'status.json')

    # 儲存使用者上傳的檔案
    file.save(input_path)
    write_status(status_path, 'pending')

    # 背景執行任務
    def background_task():
        write_status(status_path, 'running')
        try:
            scoring = ScoringCSV()
            scoring.run(input_path=input_path, output_csv_path=output_path)
            write_status(status_path, 'done')
        except Exception as e:
            write_status(status_path, 'error', str(e))

    threading.Thread(target=background_task).start()

    return jsonify({'task_id': task_id, 'status': 'pending'})

# 任務狀態查詢
@app.route('/status/<task_id>', methods=['GET'])
def status(task_id):
    status_path = os.path.join(TASK_ROOT, task_id, 'status.json')
    return jsonify(read_status(status_path))

# 結果表格（JSON 顯示用）
@app.route('/result/<task_id>', methods=['GET'])
def result(task_id):
    output_path = os.path.join(TASK_ROOT, task_id, 'output.csv')
    if not os.path.exists(output_path):
        return jsonify({'error': 'Result not found'}), 404

    import pandas as pd
    df = pd.read_csv(output_path)
    return df.to_json(orient='records')

# 結果下載（CSV 檔）
@app.route('/download/<task_id>', methods=['GET'])
def download(task_id):
    output_path = os.path.join(TASK_ROOT, task_id, 'output.csv')
    if not os.path.exists(output_path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(output_path, as_attachment=True)

# 啟動伺服器
if __name__ == '__main__':
    app.run(debug=True, port=5000)

