#!/usr/bin/env python3
"""
Vision API Server
处理来自difference_detection的webhook请求
接收base64编码的图像和ISO格式时间戳
"""

import os
import base64
import json
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

OUTPUT_DIR = None


def ensure_output_dir():
    global OUTPUT_DIR
    if OUTPUT_DIR and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


@app.route('/api/vision', methods=['POST'])
def handle_vision_event():
    """
    处理vision事件
    接收JSON数据: {"image_base64": "...", "timestamp": "..."}
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json()

        if 'image_base64' not in data or 'timestamp' not in data:
            return jsonify({"error": "Missing image_base64 or timestamp"}), 400

        image_base64 = data['image_base64']
        timestamp_str = data['timestamp']

        # 解码base64图像
        try:
            image_data = base64.b64decode(image_base64)
        except Exception as e:
            return jsonify({"error": f"Invalid base64 encoding: {str(e)}"}), 400

        # 保存图像
        ensure_output_dir()
        timestamp_dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
        filename = f"event_{timestamp_dt.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, 'wb') as f:
            f.write(image_data)

        print(f"[RECEIVED] Timestamp: {timestamp_str}, Saved: {filepath}, Size: {len(image_data)} bytes")

        return jsonify({
            "status": "success",
            "message": f"Image saved to {filepath}",
            "timestamp": timestamp_str,
            "size": len(image_data)
        }), 200

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "Vision API Server",
        "endpoints": {
            "/api/vision": "POST - Receive vision events",
            "/api/health": "GET - Health check"
        }
    }), 200


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Vision API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--output-dir', default='received_events', help='Output directory for received images')
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir
    ensure_output_dir()

    print(f"Starting Vision API Server on {args.host}:{args.port}")
    print(f"Output directory: {OUTPUT_DIR}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()