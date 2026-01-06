from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
import evaluator
import ingest

app = Flask(__name__)

UPLOAD_FOLDER = 'resumes'
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Re-ingest resumes after upload
        ingest.ingest_resumes()

        return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/rag-config', methods=['GET'])
def get_rag_config():
    """Get available RAG configuration options"""
    config = {
        'strategies': [
            {'value': 'similarity', 'label': 'Similarity Search', 'description': 'Standard semantic similarity'},
            {'value': 'mmr', 'label': 'MMR (Maximal Marginal Relevance)', 'description': 'Balance relevance and diversity'},
            {'value': 'compression', 'label': 'LLM Compression', 'description': 'Compress retrieved content with LLM'},
            {'value': 'hybrid', 'label': 'Hybrid (MMR + Compression)', 'description': 'Best of both worlds'}
        ],
        'k_values': [2, 3, 4, 5, 6, 8, 10],
        'default_strategy': 'hybrid',
        'default_k': 4,
        'default_expansion': True
    }
    return jsonify(config)

@app.route('/evaluate/<filename>', methods=['POST'])
def evaluate_single(filename):
    try:
        # Check if file exists
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'Resume file not found'}), 404

        # Get RAG parameters from request
        strategy = request.args.get('strategy', 'hybrid')
        k = int(request.args.get('k', 4))
        use_expansion = request.args.get('expansion', 'true').lower() == 'true'

        rag = evaluator.AdvancedRAG()
        result = rag.evaluate_with_rag(strategy=strategy, k=k, use_query_expansion=use_expansion)
        return jsonify({'result': result, 'filename': filename, 'strategy': strategy, 'k': k}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        # Get RAG parameters from request
        strategy = request.args.get('strategy', 'hybrid')
        k = int(request.args.get('k', 4))
        use_expansion = request.args.get('expansion', 'true').lower() == 'true'

        rag = evaluator.AdvancedRAG()
        result = rag.evaluate_with_rag(strategy=strategy, k=k, use_query_expansion=use_expansion)
        return jsonify({'result': result, 'strategy': strategy, 'k': k}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)