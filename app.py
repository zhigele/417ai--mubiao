from flask import Flask, request, jsonify, render_template_string, send_file, session, redirect, url_for
from flask_cors import CORS
import torch
import cv2
import numpy as np
import os
import uuid
import webbrowser
import threading
import functools
import hashlib
import pymysql
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import traceback

app = Flask(__name__)
app.secret_key = 'yolo_detection_secret_2024_xkq9z'

CORS(app, supports_credentials=True)

# ================= 文件配置 =================
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_MODEL_EXTENSIONS = {'pt', 'pth'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model_cache = {}

# ================= 数据库配置 =================
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'test2',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

def get_db():
    return pymysql.connect(**DB_CONFIG, connect_timeout=5, read_timeout=5, write_timeout=5)

def init_db():
    """初始化数据库，自动创建库和表，失败时打印警告但不影响服务启动"""
    conn = None
    server_conn = None
    try:
        # 先连接 MySQL 服务器（不指定 database），创建 test2 库（如果不存在）
        server_conn = pymysql.connect(
            host='localhost', port=3306, user='root',
            password='123456', charset='utf8mb4',
            connect_timeout=5, read_timeout=5, write_timeout=5
        )
        with server_conn.cursor() as cursor:
            cursor.execute("CREATE DATABASE IF NOT EXISTS test2 CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        server_conn.commit()
        server_conn.close()
        server_conn = None

        # 再连接 test2 数据库建表
        conn = get_db()
        with conn.cursor() as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(64) NOT NULL,
                    is_admin TINYINT(1) DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    model_name VARCHAR(100),
                    image_name VARCHAR(200),
                    detections INT DEFAULT 0,
                    conf_threshold FLOAT DEFAULT 0.25,
                    iou_threshold FLOAT DEFAULT 0.45,   
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_user (user_id),
                    INDEX idx_created (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            ''')
            def safe_alter(sql):
                try:
                    cursor.execute(sql)
                except Exception:
                    pass
            safe_alter("ALTER TABLE users ADD COLUMN is_admin TINYINT(1) DEFAULT 0")
            safe_alter("ALTER TABLE detection_logs ADD COLUMN conf_threshold FLOAT DEFAULT 0.25")
            safe_alter("ALTER TABLE detection_logs ADD COLUMN iou_threshold FLOAT DEFAULT 0.45")
            conn.commit()
            cursor.execute("SELECT id FROM users WHERE username='admin'")
            admin_exists = cursor.fetchone()
            if not admin_exists:
                cursor.execute("INSERT INTO users (username, password, is_admin) VALUES (%s, %s, 1)",
                              ('admin', hash_password('admin123')))
                conn.commit()
                print("✅ 已创建默认管理员账号: admin / admin123")
        print("✅ 数据库初始化成功")
    except pymysql.err.OperationalError as e:
        print(f"⚠️  MySQL 连接失败（服务仍可运行，登录功能暂时不可用）: {e}")
        print(f"   请确保 MySQL 服务已启动，且 root/123456 可连接")
    except Exception as e:
        print(f"⚠️  数据库初始化失败（服务仍可运行）: {e}")
    finally:
        if conn:
            try: conn.close()
            except Exception: pass
        if server_conn:
            try: server_conn.close()
            except Exception: pass

def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

# ================= 登录/注册模板 =================
LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv8 目标检测 - 登录</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .tab-active { border-bottom: 2px solid #3b82f6; color: #3b82f6; }
        .input-field { width:100%; padding:0.6rem 1rem; border:1px solid #d1d5db; border-radius:0.5rem; outline:none; font-size:0.95rem; transition:border-color 0.2s; }
        .input-field:focus { border-color:#3b82f6; box-shadow:0 0 0 2px rgba(59,130,246,0.15); }
        body { background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); min-height:100vh; }
    </style>
</head>
<body class="flex items-center justify-center min-h-screen">
    <div class="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-md">
        <div class="text-center mb-6">
            <div class="text-5xl mb-2">🎯</div>
            <h1 class="text-2xl font-bold text-gray-800">YOLOv8 目标检测</h1>
            <p class="text-gray-500 text-sm mt-1">请登录后使用检测功能</p>
        </div>
        <div class="flex border-b border-gray-200 mb-6">
            <button id="loginTab" onclick="switchTab('login')" class="flex-1 py-2 text-sm font-medium text-center tab-active">登录</button>
            <button id="registerTab" onclick="switchTab('register')" class="flex-1 py-2 text-sm font-medium text-center text-gray-500 hover:text-gray-700">注册</button>
        </div>
        <div id="alertBox" class="hidden mb-4 p-3 rounded-lg text-sm"></div>
        <div id="loginForm">
            <div class="mb-4">
                <label class="block text-sm font-medium text-gray-700 mb-1">用户名</label>
                <input type="text" id="loginUsername" class="input-field" placeholder="请输入用户名" autocomplete="username">
            </div>
            <div class="mb-6">
                <label class="block text-sm font-medium text-gray-700 mb-1">密码</label>
                <input type="password" id="loginPassword" class="input-field" placeholder="请输入密码" autocomplete="current-password" onkeydown="if(event.key==='Enter')doLogin()">
            </div>
            <button onclick="doLogin()" class="w-full py-2.5 bg-blue-500 hover:bg-blue-600 text-white font-semibold rounded-lg transition-colors">登录</button>
        </div>
        <div id="registerForm" class="hidden">
            <div class="mb-4">
                <label class="block text-sm font-medium text-gray-700 mb-1">用户名</label>
                <input type="text" id="regUsername" class="input-field" placeholder="4-20个字符，字母/数字/下划线">
            </div>
            <div class="mb-4">
                <label class="block text-sm font-medium text-gray-700 mb-1">密码</label>
                <input type="password" id="regPassword" class="input-field" placeholder="至少6位密码">
            </div>
            <div class="mb-6">
                <label class="block text-sm font-medium text-gray-700 mb-1">确认密码</label>
                <input type="password" id="regPasswordConfirm" class="input-field" placeholder="再次输入密码" onkeydown="if(event.key==='Enter')doRegister()">
            </div>
            <button onclick="doRegister()" class="w-full py-2.5 bg-purple-500 hover:bg-purple-600 text-white font-semibold rounded-lg transition-colors">注册</button>
        </div>
    </div>
    <script>
        function switchTab(tab) {
            const isLogin = tab === 'login';
            document.getElementById('loginForm').classList.toggle('hidden', !isLogin);
            document.getElementById('registerForm').classList.toggle('hidden', isLogin);
            document.getElementById('loginTab').className = 'flex-1 py-2 text-sm font-medium text-center ' + (isLogin ? 'tab-active' : 'text-gray-500 hover:text-gray-700');
            document.getElementById('registerTab').className = 'flex-1 py-2 text-sm font-medium text-center ' + (!isLogin ? 'tab-active' : 'text-gray-500 hover:text-gray-700');
            hideAlert();
        }
        function showAlert(msg, type) {
            const el = document.getElementById('alertBox');
            el.textContent = msg;
            el.className = 'mb-4 p-3 rounded-lg text-sm ' + (type === 'error' ? 'bg-red-50 text-red-700 border border-red-200' : 'bg-green-50 text-green-700 border border-green-200');
            el.classList.remove('hidden');
        }
        function hideAlert() { document.getElementById('alertBox').classList.add('hidden'); }
        async function doLogin() {
            const username = document.getElementById('loginUsername').value.trim();
            const password = document.getElementById('loginPassword').value;
            if (!username || !password) { showAlert('请输入用户名和密码', 'error'); return; }
            try {
                const res = await fetch('/api/login', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({username, password})});
                const data = await res.json();
                if (data.success) {
                    showAlert('登录成功，正在跳转...', 'success');
                    setTimeout(() => window.location.href = data.is_admin ? '/admin' : '/', 800);
                } else { showAlert(data.error || '登录失败', 'error'); }
            } catch(e) { showAlert('网络错误，请重试', 'error'); }
        }
        async function doRegister() {
            const username = document.getElementById('regUsername').value.trim();
            const password = document.getElementById('regPassword').value;
            const confirm = document.getElementById('regPasswordConfirm').value;
            if (!username || !password) { showAlert('请填写完整信息', 'error'); return; }
            if (!/^[a-zA-Z0-9_]{4,20}$/.test(username)) { showAlert('用户名需4-20位，只能含字母/数字/下划线', 'error'); return; }
            if (password.length < 6) { showAlert('密码至少6位', 'error'); return; }
            if (password !== confirm) { showAlert('两次密码不一致', 'error'); return; }
            try {
                const res = await fetch('/api/register', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({username, password})});
                const data = await res.json();
                if (data.success) {
                    showAlert('注册成功！请登录', 'success');
                    setTimeout(() => switchTab('login'), 1200);
                } else { showAlert(data.error || '注册失败', 'error'); }
            } catch(e) { showAlert('网络错误，请重试', 'error'); }
        }
    </script>
</body>
</html>
'''

# ================= 主检测页模板 =================
MAIN_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv8 目标检测</title>
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <style>
        *{margin:0;padding:0;box-sizing:border-box}
        body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh}
        .container{max-width:1200px;margin:0 auto;padding:20px}
        .card{background:#fff;border-radius:16px;box-shadow:0 10px 40px rgba(0,0,0,0.1);margin-bottom:20px;overflow:hidden}
        .card-header{padding:20px 24px;border-bottom:1px solid #f0f0f0;background:linear-gradient(90deg,#f8f9fa 0%,#fff 100%)}
        .card-title{font-size:18px;font-weight:600;color:#2d3748;display:flex;align-items:center;gap:10px}
        .card-body{padding:24px}
        .nav{background:rgba(255,255,255,0.95);backdrop-filter:blur(10px);box-shadow:0 2px 20px rgba(0,0,0,0.08);position:sticky;top:0;z-index:100}
        .nav-inner{max-width:1200px;margin:0 auto;padding:16px 20px;display:flex;justify-content:space-between;align-items:center}
        .nav-brand{display:flex;align-items:center;gap:10px;font-size:20px;font-weight:700;color:#4a5568}
        .nav-user{display:flex;align-items:center;gap:16px}
        .btn{display:inline-flex;align-items:center;gap:6px;padding:10px 20px;border-radius:8px;font-size:14px;font-weight:500;cursor:pointer;transition:all 0.2s;border:none}
        .btn-primary{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff}
        .btn-primary:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(102,126,234,0.4)}
        .btn-danger{background:#fff;color:#e53e3e;border:1px solid #fed7d7}
        .btn-danger:hover{background:#fff5f5}
        .btn-success{background:linear-gradient(135deg,#48bb78 0%,#38a169 100%);color:#fff}
        .upload-zone{border:2px dashed #cbd5e0;border-radius:12px;padding:40px;text-align:center;cursor:pointer;transition:all 0.3s;background:#f7fafc}
        .upload-zone:hover{border-color:#667eea;background:#edf2f7}
        .upload-zone.drag-over{border-color:#667eea;background:#e6fffa;border-style:solid}
        .upload-icon{width:48px;height:48px;margin:0 auto 16px;color:#a0aec0}
        .upload-text{color:#718096;font-size:14px}
        .upload-text span{color:#667eea;font-weight:600}
        .form-group{margin-bottom:20px}
        .form-label{display:block;font-size:14px;font-weight:500;color:#4a5568;margin-bottom:8px}
        .slider{width:100%;height:6px;border-radius:3px;background:#e2e8f0;outline:none;-webkit-appearance:none}
        .slider::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;border-radius:50%;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);cursor:pointer;box-shadow:0 2px 6px rgba(102,126,234,0.4)}
        .slider-value{text-align:center;color:#667eea;font-weight:600;font-size:14px;margin-top:8px}
        .status-bar{padding:12px 20px;border-radius:8px;margin-bottom:20px;display:flex;align-items:center;justify-content:center;gap:10px}
        .status-success{background:#c6f6d5;color:#22543d}
        .loading-spinner{width:20px;height:20px;border:2px solid #e2e8f0;border-top-color:#667eea;border-radius:50%;animation:spin 1s linear infinite}
        @keyframes spin{to{transform:rotate(360deg)}}
        .result-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}
        @media(max-width:768px){.result-grid{grid-template-columns:1fr}}
        .stat-card{background:linear-gradient(135deg,#ebf8ff 0%,#e6fffa 100%);border-radius:12px;padding:20px;text-align:center}
        .stat-value{font-size:32px;font-weight:700;color:#2b6cb0}
        .stat-label{font-size:12px;color:#718096;margin-top:4px}
        .detection-list{max-height:200px;overflow-y:auto}
        .detection-item{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:#f7fafc;border-radius:8px;margin-bottom:8px}
        .badge{padding:4px 10px;border-radius:20px;font-size:12px;font-weight:600}
        .badge-blue{background:#ebf8ff;color:#2b6cb0}
        .badge-green{background:#c6f6d5;color:#22543d}
        .hidden{display:none!important}
        .fade-in{animation:fadeIn 0.5s ease}
        @keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
        .image-preview{max-width:100%;max-height:400px;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.1)}
        .chat-btn{position:fixed;bottom:24px;right:24px;width:56px;height:56px;border-radius:50%;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;border:none;box-shadow:0 4px 20px rgba(102,126,234,0.4);cursor:pointer;font-size:24px;transition:all 0.3s;z-index:1000}
        .chat-btn:hover{transform:scale(1.1)}
        .chat-panel{position:fixed;bottom:100px;right:24px;width:380px;height:520px;background:rgba(255,255,255,0.95);backdrop-filter:blur(20px);border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,0.3);display:flex;flex-direction:column;z-index:1000;overflow:hidden;border:1px solid rgba(255,255,255,0.3)}
        .chat-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:16px 20px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
        .chat-header-info{display:flex;align-items:center;gap:12px}
        .chat-avatar{width:40px;height:40px;background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:20px;box-shadow:0 4px 12px rgba(0,0,0,0.2)}
        .chat-title{color:#fff;font-weight:700;font-size:15px}
        .chat-subtitle{color:rgba(255,255,255,0.8);font-size:12px}
        .chat-header-actions{display:flex;align-items:center;gap:8px}
        .chat-action-btn{background:rgba(255,255,255,0.2);border:none;color:#fff;padding:6px 10px;border-radius:8px;cursor:pointer;font-size:14px;transition:all 0.2s}
        .chat-action-btn:hover{background:rgba(255,255,255,0.3)}
        .chat-close-btn{background:none;border:none;color:#fff;font-size:20px;cursor:pointer;padding:4px 8px;transition:all 0.2s}
        .chat-close-btn:hover{color:rgba(255,255,255,0.7)}
        .chat-body{flex:1;overflow-y:auto;padding:16px;background:#f8f9fa}
        .chat-quick-actions{padding:12px 16px;background:#fff;border-top:1px solid #e9ecef;display:flex;gap:8px;flex-wrap:wrap;flex-shrink:0}
        .quick-btn{padding:6px 12px;background:#f0f4f8;border:1px solid #e2e8f0;border-radius:16px;font-size:12px;color:#4a5568;cursor:pointer;transition:all 0.2s}
        .quick-btn:hover{background:#e6fffa;border-color:#38b2ac;color:#319795}
        .chat-input-area{padding:12px 16px;background:#fff;border-top:1px solid #e9ecef;flex-shrink:0}
        .chat-input-wrapper{display:flex;align-items:center;gap:8px}
        .chat-input{flex:1;padding:10px 16px;border:2px solid #e2e8f0;border-radius:24px;font-size:14px;outline:none;transition:all 0.2s;background:#f8f9fa}
        .chat-input:focus{border-color:#667eea;background:#fff}
        .chat-send-btn{width:40px;height:40px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);border:none;border-radius:50%;color:#fff;font-size:16px;cursor:pointer;transition:all 0.2s;box-shadow:0 4px 12px rgba(102,126,234,0.4)}
        .chat-send-btn:hover{transform:scale(1.05)}
        .chat-msg-user{max-width:80%;padding:10px 14px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;border-radius:18px 18px 4px 18px;font-size:14px;line-height:1.5;word-wrap:break-word;align-self:flex-end;margin-left:auto}
        .chat-msg-ai{max-width:80%;padding:10px 14px;background:#fff;color:#2d3748;border-radius:18px 18px 18px 4px;font-size:14px;line-height:1.5;word-wrap:break-word;box-shadow:0 2px 8px rgba(0,0,0,0.1);align-self:flex-start}
    </style>
</head>
<body>
    <nav class="nav">
        <div class="nav-inner">
            <div class="nav-brand">
                <span>🎯</span>
                <span>YOLOv8 目标检测</span>
            </div>
            <div class="nav-user">
                <span style="color:#718096;font-size:14px">{{username}}</span>
                <button onclick="doLogout()" class="btn btn-danger">退出登录</button>
            </div>
        </div>
    </nav>
    <div class="container">
        <div id="serverStatus" class="status-bar" style="background:#fef3c7;color:#92400e">
            <div class="loading-spinner"></div>
            <span>正在连接服务器...</span>
        </div>
        <div id="modelSection" class="card hidden">
        <script>
        (function(){
            var ss=document.getElementById('serverStatus');
            if(ss){ss.className='status-bar status-success';ss.innerHTML='<span>✓ 服务就绪</span>';}
            var ms=document.getElementById('modelSection');
            if(ms){ms.style.display='block';ms.classList.remove('hidden');}
        })();
        </script>
            <div class="card-header">
                <div class="card-title">📦 步骤 1：上传模型文件</div>
            </div>
            <div class="card-body">
                <div id="modelDropZone" class="upload-zone" onclick="document.getElementById('modelInput').click()">
                    <input type="file" id="modelInput" accept=".pt,.pth" style="display:none">
                    <div id="modelUploadContent">
                        <svg class="upload-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg>
                        <p class="upload-text">点击或拖拽 <span>.pt / .pth</span> 模型文件到此处</p>
                    </div>
                    <div id="modelInfo" class="hidden" style="padding:20px">
                        <div style="width:48px;height:48px;margin:0 auto 12px;background:#c6f6d5;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#38a169;font-size:24px">✓</div>
                        <p style="color:#38a169;font-weight:600" id="modelFileName"></p>
                        <p style="color:#718096;font-size:13px;margin-top:4px" id="modelClasses"></p>
                        <button onclick="clearModel(event)" class="btn btn-danger" style="margin-top:12px">移除</button>
                    </div>
                </div>
                <div id="modelLoading" class="hidden" style="text-align:center;padding:20px">
                    <div class="loading-spinner" style="margin:0 auto 10px"></div>
                    <span style="color:#718096">正在加载模型...</span>
                </div>
            </div>
        </div>
        <div id="imageSection" class="card hidden">
            <div class="card-header">
                <div class="card-title">🖼️ 步骤 2：上传待检测图片</div>
            </div>
            <div class="card-body">
                <div id="imageDropZone" class="upload-zone" onclick="document.getElementById('imageInput').click()">
                    <input type="file" id="imageInput" accept="image/*" style="display:none">
                    <div id="imageUploadContent">
                        <svg class="upload-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>
                        <p class="upload-text">点击或拖拽图片到此处（JPG / PNG / WebP）</p>
                    </div>
                    <div id="imagePreviewContainer" class="hidden" style="text-align:center">
                        <img id="imagePreview" style="max-height:200px;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.1)">
                        <button onclick="clearImage(event)" class="btn btn-danger" style="margin-top:12px">移除</button>
                    </div>
                </div>
            </div>
        </div>
        <div id="controlSection" class="card hidden">
            <div class="card-header">
                <div class="card-title">⚙️ 步骤 3：开始检测</div>
            </div>
            <div class="card-body">
                <div style="display:flex;flex-wrap:wrap;gap:24px;align-items:flex-end">
                    <div style="flex:1;min-width:200px">
                        <label class="form-label">置信度阈值</label>
                        <input type="range" id="confThreshold" min="0.05" max="0.95" step="0.05" value="0.25" class="slider" oninput="document.getElementById('confValue').textContent=this.value">
                        <div class="slider-value"><span id="confValue">0.25</span></div>
                    </div>
                    <div style="flex:1;min-width:200px">
                        <label class="form-label">IoU 阈值</label>
                        <input type="range" id="iouThreshold" min="0.05" max="0.95" step="0.05" value="0.45" class="slider" oninput="document.getElementById('iouValue').textContent=this.value">
                        <div class="slider-value"><span id="iouValue">0.45</span></div>
                    </div>
                    <button id="detectBtn" onclick="startDetection()" class="btn btn-primary" style="padding:12px 32px;font-size:16px">🚀 开始检测</button>
                </div>
            </div>
        </div>
        <div id="resultSection" class="card hidden">
            <div class="card-header">
                <div class="card-title">📊 检测结果</div>
            </div>
            <div class="card-body">
                <div id="detectionLoading" class="hidden" style="text-align:center;padding:60px">
                    <div class="loading-spinner" style="width:40px;height:40px;margin:0 auto 16px"></div>
                    <span style="color:#a0aec0">正在检测，请稍候...</span>
                </div>
                <div id="detectionResult" class="hidden">
                    <div class="result-grid">
                        <div>
                            <p style="font-size:12px;color:#a0aec0;margin-bottom:8px">📷 检测可视化</p>
                            <div id="resultImgWrapper" style="background:#f7fafc;border-radius:12px;padding:16px;min-height:320px;display:flex;align-items:center;justify:center;border:1px solid #e2e8f0">
                                <div id="resultImgLoading" style="text-align:center;color:#cbd5e0">
                                    <div style="font-size:48px;margin-bottom:8px">⏳</div>
                                    <p>正在渲染图片...</p>
                                </div>
                                <img id="resultImage" class="hidden image-preview" alt="检测结果">
                            </div>
                            <button onclick="downloadResult()" class="btn btn-success" style="width:100%;margin-top:16px">⬇️ 下载结果图片</button>
                        </div>
                        <div style="display:flex;flex-direction:column;gap:16px">
                            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
                                <div class="stat-card">
                                    <div class="stat-value" id="totalDetections">0</div>
                                    <div class="stat-label">检测目标数</div>
                                </div>
                                <div class="stat-card" style="background:linear-gradient(135deg,#c6f6d5 0%,#9ae6b4 100%)">
                                    <div class="stat-value" style="color:#276749" id="processingTime">0ms</div>
                                    <div class="stat-label">处理时间</div>
                                </div>
                            </div>
                            <div style="background:#f7fafc;border-radius:12px;padding:16px;flex:1">
                                <p style="font-size:12px;color:#a0aec0;margin-bottom:12px">📈 类别统计</p>
                                <div id="detectionList" class="detection-list"></div>
                                <p id="noDetectionTip" class="hidden" style="text-align:center;color:#cbd5e0;padding:20px">未检测到任何目标</p>
                            </div>
                            <div style="background:#f7fafc;border-radius:12px;padding:16px;flex:1">
                                <p style="font-size:12px;color:#a0aec0;margin-bottom:12px">📋 检测详情</p>
                                <div id="detectionDetails" class="detection-list"></div>
                                <p id="noDetailsTip" class="hidden" style="text-align:center;color:#cbd5e0;padding:20px">暂无详情</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        var modelId = null, currentImage = null, resultImageData = null;

        function doLogout() { fetch('/api/logout', {method:'POST'}).then(function(){ window.location.href = '/login'; }).catch(function(){ window.location.href = '/login'; }); }

        // 健康检查：仅用于更新状态栏版本信息，不影响主界面显示
        setTimeout(function() {
            var ctrl = new AbortController();
            var tid = setTimeout(function(){ ctrl.abort(); }, 4000);
            fetch('/api/health', {signal: ctrl.signal})
                .then(function(res){ clearTimeout(tid); if(!res.ok) return null; return res.json(); })
                .then(function(d){
                    if(d && d.status==='running'){
                        var ss = document.getElementById('serverStatus');
                        if(ss) {
                            ss.style.cssText = 'margin-bottom:24px;padding:12px 16px;border-radius:8px;text-align:center;background:#f0fdf4;border:1px solid #86efac;color:#15803d;font-size:14px';
                            ss.innerHTML = '✅ 服务就绪 &nbsp;|&nbsp; PyTorch '+d.torch_version+' &nbsp;|&nbsp; CUDA: '+(d.cuda_available?'<span style="color:#16a34a">可用</span>':'<span style="color:#dc2626">不可用</span>');
                        }
                    }
                })
                .catch(function(){ clearTimeout(tid); });
        }, 800);
        var modelDZ = document.getElementById('modelDropZone');
        if(modelDZ){
            modelDZ.addEventListener('dragenter', function(e){ e.preventDefault(); e.stopPropagation(); modelDZ.classList.add('drag-over'); });
            modelDZ.addEventListener('dragover', function(e){ e.preventDefault(); e.stopPropagation(); modelDZ.classList.add('drag-over'); });
            modelDZ.addEventListener('dragleave', function(e){ e.preventDefault(); e.stopPropagation(); modelDZ.classList.remove('drag-over'); });
            modelDZ.addEventListener('drop', function(e){ e.preventDefault(); e.stopPropagation(); modelDZ.classList.remove('drag-over'); if(e.dataTransfer.files.length) handleModelFile(e.dataTransfer.files[0]); });
        }
        var modelInput = document.getElementById('modelInput');
        if(modelInput) modelInput.addEventListener('change', function(e){ if(e.target.files.length) handleModelFile(e.target.files[0]); });
        async function handleModelFile(file) {
            if(!file.name.endsWith('.pt') && !file.name.endsWith('.pth')){ alert('请上传 .pt 或 .pth 格式'); return; }
            document.getElementById('modelLoading').classList.remove('hidden');
            document.getElementById('modelUploadContent').classList.add('hidden');
            const fd = new FormData(); fd.append('model', file);
            try {
                const res = await fetch('/api/load_model', {method:'POST', body:fd}); const d = await res.json();
                if(d.success){
                    modelId = d.model_id;
                    document.getElementById('modelInfo').classList.remove('hidden');
                    document.getElementById('modelFileName').textContent = '✅ ' + d.model_name;
                    document.getElementById('modelClasses').textContent = '已识别 ' + Object.keys(d.classes).length + ' 个类别';
                    document.getElementById('imageSection').classList.remove('hidden');
                    document.getElementById('controlSection').classList.remove('hidden');
                } else { alert('模型加载失败：' + d.error); document.getElementById('modelUploadContent').classList.remove('hidden'); }
            } catch(e){ alert('上传失败：' + e.message); document.getElementById('modelUploadContent').classList.remove('hidden'); }
            document.getElementById('modelLoading').classList.add('hidden');
        }
        function clearModel(e){ e&&e.stopPropagation(); if(modelId) fetch('/api/unload_model',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model_id:modelId})}); modelId=null; document.getElementById('modelUploadContent').classList.remove('hidden'); document.getElementById('modelInfo').classList.add('hidden'); ['imageSection','controlSection','resultSection'].forEach(id=>document.getElementById(id).classList.add('hidden')); document.getElementById('modelInput').value=''; }
        var imageDZ = document.getElementById('imageDropZone');
        if(imageDZ){
            imageDZ.addEventListener('dragenter', function(e){ e.preventDefault(); e.stopPropagation(); imageDZ.classList.add('drag-over'); });
            imageDZ.addEventListener('dragover', function(e){ e.preventDefault(); e.stopPropagation(); imageDZ.classList.add('drag-over'); });
            imageDZ.addEventListener('dragleave', function(e){ e.preventDefault(); e.stopPropagation(); imageDZ.classList.remove('drag-over'); });
            imageDZ.addEventListener('drop', function(e){ e.preventDefault(); e.stopPropagation(); imageDZ.classList.remove('drag-over'); if(e.dataTransfer.files.length) handleImageFile(e.dataTransfer.files[0]); });
        }
        var imageInput = document.getElementById('imageInput');
        if(imageInput) imageInput.addEventListener('change', function(e){ if(e.target.files.length) handleImageFile(e.target.files[0]); });
        function handleImageFile(file){
            if(!file.type.startsWith('image/')){ alert('请选择图片文件'); return; }
            currentImage = file;
            const reader = new FileReader();
            reader.onload = ev=>{ document.getElementById('imageUploadContent').classList.add('hidden'); document.getElementById('imagePreviewContainer').classList.remove('hidden'); document.getElementById('imagePreview').src=ev.target.result; };
            reader.readAsDataURL(file);
        }
        function clearImage(e){ e&&e.stopPropagation(); currentImage=null; document.getElementById('imageUploadContent').classList.remove('hidden'); document.getElementById('imagePreviewContainer').classList.add('hidden'); document.getElementById('imageInput').value=''; }
        async function startDetection(){
            if(!modelId||!currentImage){ alert('请先上传模型和图片'); return; }
            document.getElementById('resultSection').classList.remove('hidden');
            document.getElementById('detectionLoading').classList.remove('hidden');
            document.getElementById('detectionResult').classList.add('hidden');
            const t0 = performance.now();
            const fd = new FormData();
            fd.append('model_id', modelId); fd.append('image', currentImage);
            fd.append('conf_threshold', document.getElementById('confThreshold').value);
            fd.append('iou_threshold', document.getElementById('iouThreshold').value);
            try {
                const res = await fetch('/api/detect', {method:'POST', body:fd});
                if (!res.ok) {
                    if (res.status === 401) { alert('登录已失效，请重新登录'); window.location.href = '/login'; return; }
                    const errText = await res.text();
                    alert('检测失败：' + (errText.length > 200 ? errText.substring(0,200) + '...' : errText));
                    document.getElementById('detectionLoading').classList.add('hidden');
                    return;
                }
                const d = await res.json();
                const elapsed = Math.round(performance.now() - t0);
                if(d.success){
                    // 图片 - 先显示loading，再渲染
                    const imgEl = document.getElementById('resultImage');
                    const loadingEl = document.getElementById('resultImgLoading');
                    imgEl.classList.add('hidden');
                    if(loadingEl) loadingEl.classList.remove('hidden');
                    imgEl.onload = function(){
                        if(loadingEl) loadingEl.classList.add('hidden');
                        imgEl.classList.remove('hidden');
                    };
                    imgEl.onerror = function(){
                        if(loadingEl) loadingEl.innerHTML = '<div class="text-center text-red-300"><div class="text-4xl mb-2">⚠️</div><p class="text-sm">图片渲染失败</p></div>';
                        imgEl.classList.add('hidden');
                    };
                    imgEl.src = d.image;
                    resultImageData = d.image;
                    // 统计
                    document.getElementById('totalDetections').textContent = d.total_detections;
                    document.getElementById('processingTime').textContent = elapsed + 'ms';
                    // 类别统计
                    const listEl = document.getElementById('detectionList');
                    const tipEl = document.getElementById('noDetectionTip');
                    listEl.innerHTML = '';
                    if(Object.keys(d.class_counts || {}).length > 0){
                        tipEl.classList.add('hidden');
                        Object.entries(d.class_counts).forEach(([cls,cnt])=>{
                            listEl.innerHTML += '<div class="flex justify-between items-center bg-white rounded px-3 py-2 text-sm shadow-sm"><span class="text-gray-700 font-medium">'+cls+'</span><span class="font-bold text-blue-600">× '+cnt+'</span></div>';
                        });
                    } else {
                        tipEl.classList.remove('hidden');
                    }
                    // 详细列表
                    const detailEl = document.getElementById('detectionDetails');
                    const noDetailEl = document.getElementById('noDetailsTip');
                    detailEl.innerHTML = '';
                    if(d.detections && d.detections.length > 0){
                        noDetailEl.classList.add('hidden');
                        d.detections.forEach((det, i)=>{
                            detailEl.innerHTML += '<div class="bg-white rounded px-3 py-2 text-xs shadow-sm"><span class="font-semibold text-blue-600">'+det.class_name+'</span> <span class="text-gray-400">置信度</span> <span class="font-bold text-green-600">'+(det.confidence*100).toFixed(1)+'%</span></div>';
                        });
                    } else {
                        noDetailEl.classList.remove('hidden');
                    }
                    document.getElementById('detectionResult').classList.remove('hidden');
                } else {
                    alert('检测失败：' + (d.error || '未知错误'));
                }
            } catch(e){ console.error('检测异常:', e); alert('检测出错：' + e.message); }
            document.getElementById('detectionLoading').classList.add('hidden');
        }
        function downloadResult(){ if(resultImageData){ const a=document.createElement('a'); a.download='yolo_result.jpg'; a.href=resultImageData; a.click(); } }

        // ====== AI 助手聊天 ======
        const CHAT_KEY = 'yolo_chat_history';
        let chatOpen = false;

        function getHistory(){ try{ return JSON.parse(localStorage.getItem(CHAT_KEY)||'[]'); } catch{ return []; } }
        function saveHistory(h){ localStorage.setItem(CHAT_KEY, JSON.stringify(h)); }
        function scrollBottom(){ setTimeout(()=>{ const el=document.getElementById('chatBody'); if(el) el.scrollTop=el.scrollHeight; }, 50); }

        function renderChat(){
            var h = getHistory();
            var body = document.getElementById('chatBody');
            body.innerHTML = h.map(function(m){
                var isUser = m.role==='user';
                var content = m.content.replace(/\\n/g,'<br>');
                return '<div style="display:flex;margin-bottom:12px;'+(isUser?'justify-content:flex-end':'justify-content:flex-start')+'">'+
                    (isUser?'':'<div style="width:32px;height:32px;background:linear-gradient(135deg,#667eea,#764ba2);border-radius:50%;display:flex;align-items:center;justify-content:center;color:#fff;font-size:12px;font-weight:bold;margin-right:8px;flex-shrink:0">AI</div>')+
                    '<div class="'+(isUser?'chat-msg-user':'chat-msg-ai')+'">'+content+'</div>'+
                    (isUser?'<div style="width:32px;height:32px;background:#e2e8f0;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#718096;font-size:12px;font-weight:bold;margin-left:8px;flex-shrink:0">U</div>':'')+
                '</div>';
            }).join('');
            scrollBottom();
        }

        function toggleChat(){
            chatOpen = !chatOpen;
            const panel = document.getElementById('chatPanel');
            const btn = document.getElementById('chatToggleBtn');
            if(chatOpen){
                panel.classList.remove('hidden');
                btn.innerHTML = '✕';
                renderChat();
            } else {
                panel.classList.add('hidden');
                btn.innerHTML = '💬';
            }
        }

        async function sendMsg(){
            const input = document.getElementById('chatInput');
            const text = input.value.trim();
            if(!text) return;
            const h = getHistory();
            h.push({role:'user', content:text});
            saveHistory(h);
            renderChat();
            input.value = '';
            document.getElementById('chatBody').innerHTML += '<div id="typingIndicator" class="flex justify-start mb-3"><div class="w-7 h-7 rounded-full bg-blue-500 flex items-center justify-center text-white text-xs font-bold mr-2">AI</div><div class="max-w-[75%] px-4 py-2.5 rounded-2xl rounded-tl-sm bg-gray-100 text-gray-400 text-sm">思考中...</div></div>';
            scrollBottom();
            try {
                const res = await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});
                const d = await res.json();
                document.getElementById('typingIndicator')?.remove();
                const h2 = getHistory();
                h2.push({role:'assistant', content: d.reply});
                saveHistory(h2);
                renderChat();
            } catch(e){
                document.getElementById('typingIndicator')?.remove();
                const h2 = getHistory();
                h2.push({role:'assistant', content:'抱歉，网络出现了问题，请稍后重试。'});
                saveHistory(h2);
                renderChat();
            }
        }

        function clearChat(){
            localStorage.removeItem(CHAT_KEY);
            renderChat();
        }
    </script>

    <!-- AI 助手悬浮按钮 -->
    <button id="chatToggleBtn" onclick="toggleChat()" class="chat-btn">💬</button>

    <!-- AI 助手聊天面板 -->
    <div id="chatPanel" class="chat-panel hidden">
        <!-- 头部 -->
        <div class="chat-header">
            <div class="chat-header-info">
                <div class="chat-avatar">🤖</div>
                <div>
                    <p class="chat-title">AI 助手</p>
                    <p class="chat-subtitle">YOLOv8 & 目标检测专家</p>
                </div>
            </div>
            <div class="chat-header-actions">
                <button onclick="clearChat()" class="chat-action-btn" title="清空记录">🗑️</button>
                <button onclick="toggleChat()" class="chat-close-btn">✕</button>
            </div>
        </div>
        <!-- 消息区域 -->
        <div id="chatBody" class="chat-body"></div>
        <!-- 快捷问题 -->
        <div class="chat-quick-actions">
            <button onclick="document.getElementById('chatInput').value='怎么上传模型？';sendMsg();" class="quick-btn">📦 上传模型</button>
            <button onclick="document.getElementById('chatInput').value='置信度阈值怎么调？';sendMsg();" class="quick-btn">⚙️ 置信度设置</button>
            <button onclick="document.getElementById('chatInput').value='IoU阈值是什么？';sendMsg();" class="quick-btn">📐 IoU阈值</button>
            <button onclick="document.getElementById('chatInput').value='检测不到目标怎么办？';sendMsg();" class="quick-btn">🔍 排查问题</button>
        </div>
        <!-- 输入区域 -->
        <div class="chat-input-area">
            <div class="chat-input-wrapper">
                <input type="text" id="chatInput" class="chat-input" placeholder="输入问题，按 Enter 发送..." onkeydown="if(event.key=='Enter')sendMsg()">
                <button onclick="sendMsg()" class="chat-send-btn">➤</button>
            </div>
        </div>
    </div>
</body>
</html>
'''

# ================= 管理后台模板 =================
ADMIN_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>管理后台 - YOLOv8 目标检测</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .loading-spinner { border:3px solid #f3f3f3; border-top:3px solid #3b82f6; border-radius:50%; width:32px; height:32px; animation:spin 1s linear infinite; }
        @keyframes spin { 0%{transform:rotate(0deg)} 100%{transform:rotate(360deg)} }
        .tab-active { border-bottom:2px solid #3b82f6; color:#3b82f6; }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <nav class="bg-gray-900 shadow-lg sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4">
            <div class="flex items-center justify-between h-14">
                <div class="flex items-center space-x-3">
                    <span class="text-xl">🛡️</span>
                    <span class="text-white font-bold text-lg">管理后台</span>
                    <span class="text-xs px-2 py-0.5 bg-red-500 text-white rounded-full">ADMIN</span>
                </div>
                <div class="flex items-center space-x-4">
                    <a href="/" class="text-gray-300 hover:text-white text-sm px-3 py-1.5 rounded hover:bg-gray-700 transition-colors">检测页面</a>
                    <button onclick="doLogout()" class="text-gray-300 hover:text-red-400 text-sm px-3 py-1.5 rounded hover:bg-gray-700 transition-colors">退出登录</button>
                </div>
            </div>
        </div>
    </nav>

    <div class="max-w-7xl mx-auto px-4 py-8">
        <!-- 统计卡片 -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="bg-white rounded-xl shadow p-6">
                <div class="flex items-center justify-between">
                    <div>
                        <div class="text-sm text-gray-500">总用户数</div>
                        <div id="statUsers" class="text-3xl font-bold text-blue-600 mt-1">-</div>
                    </div>
                    <div class="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center text-2xl">👥</div>
                </div>
            </div>
            <div class="bg-white rounded-xl shadow p-6">
                <div class="flex items-center justify-between">
                    <div>
                        <div class="text-sm text-gray-500">总检测次数</div>
                        <div id="statDetections" class="text-3xl font-bold text-green-600 mt-1">-</div>
                    </div>
                    <div class="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center text-2xl">🔍</div>
                </div>
            </div>
            <div class="bg-white rounded-xl shadow p-6">
                <div class="flex items-center justify-between">
                    <div>
                        <div class="text-sm text-gray-500">今日检测</div>
                        <div id="statToday" class="text-3xl font-bold text-purple-600 mt-1">-</div>
                    </div>
                    <div class="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center text-2xl">📅</div>
                </div>
            </div>
            <div class="bg-white rounded-xl shadow p-6">
                <div class="flex items-center justify-between">
                    <div>
                        <div class="text-sm text-gray-500">管理员数</div>
                        <div id="statAdmins" class="text-3xl font-bold text-orange-600 mt-1">-</div>
                    </div>
                    <div class="w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center text-2xl">⭐</div>
                </div>
            </div>
        </div>

        <!-- 标签页 -->
        <div class="bg-white rounded-xl shadow mb-6">
            <div class="flex border-b border-gray-200">
                <button id="tabUsers" onclick="switchTab('users')" class="px-6 py-3 text-sm font-medium text-center tab-active">用户管理</button>
                <button id="tabLogs" onclick="switchTab('logs')" class="px-6 py-3 text-sm font-medium text-center text-gray-500 hover:text-gray-700">检测记录</button>
                <button id="tabSettings" onclick="switchTab('settings')" class="px-6 py-3 text-sm font-medium text-center text-gray-500 hover:text-gray-700">系统设置</button>
            </div>

            <!-- 用户管理 -->
            <div id="panelUsers" class="p-6">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-lg font-semibold text-gray-700">用户列表</h3>
                    <button onclick="loadUsers()" class="text-sm text-blue-500 hover:text-blue-700">🔄 刷新</button>
                </div>
                <div id="usersLoading" class="flex items-center justify-center py-12"><div class="loading-spinner"></div><span class="ml-3 text-gray-500">加载中...</span></div>
                <div id="usersAlert" class="hidden mb-4 p-3 rounded-lg text-sm"></div>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="bg-gray-50 text-gray-600">
                                <th class="px-4 py-3 text-left font-medium">ID</th>
                                <th class="px-4 py-3 text-left font-medium">用户名</th>
                                <th class="px-4 py-3 text-left font-medium">角色</th>
                                <th class="px-4 py-3 text-left font-medium">注册时间</th>
                                <th class="px-4 py-3 text-left font-medium">检测次数</th>
                                <th class="px-4 py-3 text-center font-medium">操作</th>
                            </tr>
                        </thead>
                        <tbody id="usersTable" class="divide-y divide-gray-100"></tbody>
                    </table>
                </div>
            </div>

            <!-- 检测记录 -->
            <div id="panelLogs" class="p-6 hidden">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-lg font-semibold text-gray-700">检测记录</h3>
                    <button onclick="loadLogs()" class="text-sm text-blue-500 hover:text-blue-700">🔄 刷新</button>
                </div>
                <div id="logsLoading" class="flex items-center justify-center py-12"><div class="loading-spinner"></div><span class="ml-3 text-gray-500">加载中...</span></div>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="bg-gray-50 text-gray-600">
                                <th class="px-4 py-3 text-left font-medium">ID</th>
                                <th class="px-4 py-3 text-left font-medium">用户</th>
                                <th class="px-4 py-3 text-left font-medium">模型</th>
                                <th class="px-4 py-3 text-left font-medium">图片</th>
                                <th class="px-4 py-3 text-center font-medium">检测数</th>
                                <th class="px-4 py-3 text-left font-medium">时间</th>
                            </tr>
                        </thead>
                        <tbody id="logsTable" class="divide-y divide-gray-100"></tbody>
                    </table>
                </div>
            </div>

            <!-- 系统设置 -->
            <div id="panelSettings" class="p-6 hidden">
                <h3 class="text-lg font-semibold text-gray-700 mb-4">系统设置</h3>
                <div id="settingsAlert" class="hidden mb-4 p-3 rounded-lg text-sm"></div>
                <div class="space-y-6 max-w-lg">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">当前管理员密码修改</label>
                        <input type="password" id="newPassword" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500" placeholder="输入新密码（留空则不修改）">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">添加新管理员</label>
                        <div class="flex gap-2">
                            <input type="text" id="newAdminUsername" class="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500" placeholder="用户名">
                            <input type="password" id="newAdminPassword" class="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500" placeholder="密码">
                            <button onclick="addAdmin()" class="px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors text-sm">添加</button>
                        </div>
                    </div>
                    <div>
                        <button onclick="saveSettings()" class="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors">💾 保存设置</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function switchTab(tab) {
            ['users','logs','settings'].forEach(t => {
                document.getElementById('tab' + t.charAt(0).toUpperCase() + t.slice(1)).className = 'px-6 py-3 text-sm font-medium text-center ' + (t===tab ? 'tab-active' : 'text-gray-500 hover:text-gray-700');
                document.getElementById('panel' + t.charAt(0).toUpperCase() + t.slice(1)).classList.toggle('hidden', t!==tab);
            });
            if(tab==='users') loadUsers();
            if(tab==='logs') loadLogs();
        }

        async function doLogout() { await fetch('/api/logout',{method:'POST'}); window.location.href='/login'; }

        async function loadStats() {
            try {
                const res = await fetch('/api/admin/stats');
                const d = await res.json();
                if(d.success){
                    document.getElementById('statUsers').textContent = d.stats.total_users;
                    document.getElementById('statDetections').textContent = d.stats.total_detections;
                    document.getElementById('statToday').textContent = d.stats.today_detections;
                    document.getElementById('statAdmins').textContent = d.stats.admin_count;
                }
            } catch(e){ console.error(e); }
        }

        async function loadUsers() {
            document.getElementById('usersLoading').classList.remove('hidden');
            try {
                const res = await fetch('/api/admin/users');
                const d = await res.json();
                document.getElementById('usersLoading').classList.add('hidden');
                if(d.success){
                    const tbody = document.getElementById('usersTable');
                    tbody.innerHTML = d.users.map(u => `
                        <tr class="hover:bg-gray-50">
                            <td class="px-4 py-3 text-gray-500">${u.id}</td>
                            <td class="px-4 py-3 font-medium">${u.username}</td>
                            <td class="px-4 py-3">${u.is_admin ? '<span class="px-2 py-0.5 bg-orange-100 text-orange-700 rounded text-xs font-medium">管理员</span>' : '<span class="px-2 py-0.5 bg-gray-100 text-gray-600 rounded text-xs">普通用户</span>'}</td>
                            <td class="px-4 py-3 text-gray-500">${u.created_at}</td>
                            <td class="px-4 py-3 text-center font-medium text-blue-600">${u.detection_count || 0}</td>
                            <td class="px-4 py-3 text-center">
                                ${!u.is_admin ? `<button onclick="deleteUser(${u.id},'${u.username}')" class="text-red-500 hover:text-red-700 text-xs">🗑️ 删除</button>` : '-'}
                            </td>
                        </tr>
                    `).join('');
                }
            } catch(e){ document.getElementById('usersLoading').classList.add('hidden'); }
        }

        async function deleteUser(id, username) {
            if(!confirm('确定要删除用户 "' + username + '" 吗？此操作不可恢复！')) return;
            const res = await fetch('/api/admin/delete_user', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({user_id: id})});
            const d = await res.json();
            const el = document.getElementById('usersAlert');
            if(d.success){
                el.className = 'mb-4 p-3 rounded-lg text-sm bg-green-50 text-green-700 border border-green-200'; el.textContent = '✅ ' + d.message;
                loadUsers(); loadStats();
            } else {
                el.className = 'mb-4 p-3 rounded-lg text-sm bg-red-50 text-red-700 border border-red-200'; el.textContent = '❌ ' + d.error;
            }
            el.classList.remove('hidden');
            setTimeout(()=>el.classList.add('hidden'), 3000);
        }

        async function loadLogs() {
            document.getElementById('logsLoading').classList.remove('hidden');
            try {
                const res = await fetch('/api/admin/logs');
                const d = await res.json();
                document.getElementById('logsLoading').classList.add('hidden');
                if(d.success){
                    const tbody = document.getElementById('logsTable');
                    tbody.innerHTML = d.logs.map(l => `
                        <tr class="hover:bg-gray-50">
                            <td class="px-4 py-3 text-gray-500">${l.id}</td>
                            <td class="px-4 py-3 font-medium">${l.username}</td>
                            <td class="px-4 py-3 text-gray-600 truncate max-w-[150px]">${l.model_name || '-'}</td>
                            <td class="px-4 py-3 text-gray-600 truncate max-w-[150px]">${l.image_name || '-'}</td>
                            <td class="px-4 py-3 text-center"><span class="px-2 py-0.5 bg-blue-50 text-blue-600 rounded font-medium">${l.detections}</span></td>
                            <td class="px-4 py-3 text-gray-500 text-xs">${l.created_at}</td>
                        </tr>
                    `).join('');
                    if(!d.logs.length) tbody.innerHTML = '<tr><td colspan="6" class="px-4 py-12 text-center text-gray-400">暂无检测记录</td></tr>';
                }
            } catch(e){ document.getElementById('logsLoading').classList.add('hidden'); }
        }

        async function saveSettings() {
            const newPassword = document.getElementById('newPassword').value.trim();
            const res = await fetch('/api/admin/change_password', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({new_password: newPassword})});
            const d = await res.json();
            const el = document.getElementById('settingsAlert');
            el.className = 'mb-4 p-3 rounded-lg text-sm ' + (d.success ? 'bg-green-50 text-green-700 border border-green-200' : 'bg-red-50 text-red-700 border border-red-200');
            el.textContent = d.success ? '✅ ' + d.message : '❌ ' + d.error;
            el.classList.remove('hidden');
            if(d.success) document.getElementById('newPassword').value = '';
            setTimeout(()=>el.classList.add('hidden'), 3000);
        }

        async function addAdmin() {
            const username = document.getElementById('newAdminUsername').value.trim();
            const password = document.getElementById('newAdminPassword').value;
            if(!username || !password){ alert('请填写用户名和密码'); return; }
            if(password.length < 6){ alert('密码至少6位'); return; }
            const res = await fetch('/api/register', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({username, password, is_admin: 1})});
            const d = await res.json();
            const el = document.getElementById('settingsAlert');
            if(d.success){
                el.className = 'mb-4 p-3 rounded-lg text-sm bg-green-50 text-green-700 border border-green-200';
                el.textContent = '✅ 管理员 ' + username + ' 创建成功！';
                document.getElementById('newAdminUsername').value = '';
                document.getElementById('newAdminPassword').value = '';
                loadUsers(); loadStats();
            } else {
                el.className = 'mb-4 p-3 rounded-lg text-sm bg-red-50 text-red-700 border border-red-200';
                el.textContent = '❌ ' + d.error;
            }
            el.classList.remove('hidden');
            setTimeout(()=>el.classList.add('hidden'), 3000);
        }

        loadStats();

        // ====== AI 助手聊天 ======
        const CHAT_KEY = 'yolo_chat_history';
        let chatOpen = false;
        function getHistory(){ try{ return JSON.parse(localStorage.getItem(CHAT_KEY)||'[]'); } catch{ return []; } }
        function saveHistory(h){ localStorage.setItem(CHAT_KEY, JSON.stringify(h)); }
        function renderChat(){
            var h = getHistory();
            var body = document.getElementById('chatBody');
            if(!body) return;
            body.innerHTML = h.map(function(m){
                var isUser = m.role==='user';
                var content = m.content.replace(/\\n/g,'<br>');
                return '<div style="display:flex;margin-bottom:12px;'+(isUser?'justify-content:flex-end':'justify-content:flex-start')+'">'+
                    (isUser?'':'<div style="width:32px;height:32px;background:linear-gradient(135deg,#667eea,#764ba2);border-radius:50%;display:flex;align-items:center;justify-content:center;color:#fff;font-size:12px;font-weight:bold;margin-right:8px;flex-shrink:0">AI</div>')+
                    '<div class="'+(isUser?'chat-msg-user':'chat-msg-ai')+'">'+content+'</div>'+
                    (isUser?'<div style="width:32px;height:32px;background:#e2e8f0;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#718096;font-size:12px;font-weight:bold;margin-left:8px;flex-shrink:0">U</div>':'')+
                '</div>';
            }).join('');
            setTimeout(function(){ var el=document.getElementById('chatBody'); if(el) el.scrollTop=el.scrollHeight; }, 50);
        }
        function toggleChat(){
            chatOpen = !chatOpen;
            document.getElementById('chatPanel').classList.toggle('hidden', !chatOpen);
            document.getElementById('chatToggleBtn').innerHTML = chatOpen ? '✕' : '💬';
            if(chatOpen) renderChat();
        }
        async function sendMsg(){
            const text = document.getElementById('chatInput').value.trim();
            if(!text) return;
            const h = getHistory(); h.push({role:'user', content:text}); saveHistory(h); renderChat();
            document.getElementById('chatInput').value = '';
            document.getElementById('chatBody').innerHTML += '<div id="typingIndicator" class="flex justify-start mb-3"><div class="w-7 h-7 rounded-full bg-blue-500 flex items-center justify-center text-white text-xs font-bold mr-2">AI</div><div class="max-w-[75%] px-4 py-2.5 rounded-2xl rounded-tl-sm bg-gray-100 text-gray-400 text-sm">思考中...</div></div>';
            setTimeout(()=>{ const el=document.getElementById('chatBody'); if(el) el.scrollTop=el.scrollHeight; }, 50);
            try {
                const res = await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});
                const d = await res.json();
                document.getElementById('typingIndicator')?.remove();
                const h2 = getHistory(); h2.push({role:'assistant', content:d.reply}); saveHistory(h2); renderChat();
            } catch(e){
                document.getElementById('typingIndicator')?.remove();
                const h2 = getHistory(); h2.push({role:'assistant', content:'抱歉，网络出现问题，请稍后重试。'}); saveHistory(h2); renderChat();
            }
        }
        function clearChat(){ localStorage.removeItem(CHAT_KEY); renderChat(); }
    </script>

    <!-- AI 助手 -->
    <button id="chatToggleBtn" onclick="toggleChat()" class="fixed bottom-6 right-6 w-14 h-14 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-full shadow-lg hover:shadow-xl transition-all text-2xl z-50 flex items-center justify-center">💬</button>
    <div id="chatPanel" class="hidden fixed bottom-24 right-6 w-[380px] h-[520px] bg-white rounded-2xl shadow-2xl flex flex-col z-50 border border-gray-200 overflow-hidden">
        <div class="bg-gradient-to-r from-blue-500 to-purple-600 px-4 py-3 flex items-center justify-between flex-shrink-0">
            <div class="flex items-center space-x-2">
                <span class="text-xl">🤖</span><div><div class="text-white font-semibold text-sm">AI 助手</div><div class="text-blue-100 text-xs">YOLOv8 & 目标检测专家</div></div>
            </div>
            <div class="flex items-center space-x-2">
                <button onclick="clearChat()" class="text-blue-200 hover:text-white text-xs" title="清空记录">🗑️</button>
                <button onclick="toggleChat()" class="text-white hover:text-gray-200 text-lg leading-none">✕</button>
            </div>
        </div>
        <div id="chatBody" class="flex-1 overflow-y-auto p-4 space-y-1"></div>
        <div class="px-3 pb-2 flex gap-1.5 flex-wrap flex-shrink-0">
            <button onclick="document.getElementById('chatInput').value='怎么上传模型？';sendMsg();" class="text-xs px-2.5 py-1 bg-gray-100 hover:bg-blue-50 text-gray-600 hover:text-blue-600 rounded-full transition-colors">📦 上传模型</button>
            <button onclick="document.getElementById('chatInput').value='怎么添加管理员？';sendMsg();" class="text-xs px-2.5 py-1 bg-gray-100 hover:bg-blue-50 text-gray-600 hover:text-blue-600 rounded-full transition-colors">👤 添加管理员</button>
            <button onclick="document.getElementById('chatInput').value='检测记录在哪看？';sendMsg();" class="text-xs px-2.5 py-1 bg-gray-100 hover:bg-blue-50 text-gray-600 hover:text-blue-600 rounded-full transition-colors">📋 检测记录</button>
            <button onclick="document.getElementById('chatInput').value='怎么导出模型？';sendMsg();" class="text-xs px-2.5 py-1 bg-gray-100 hover:bg-blue-50 text-gray-600 hover:text-blue-600 rounded-full transition-colors">📤 导出模型</button>
        </div>
        <div class="border-t border-gray-100 p-3 flex-shrink-0">
            <div class="flex items-center space-x-2">
                <input type="text" id="chatInput" class="flex-1 px-3 py-2 border border-gray-200 rounded-full text-sm focus:outline-none focus:border-blue-400" placeholder="输入问题，按 Enter 发送..." onkeydown="if(event.key==='Enter')sendMsg()">
                <button onclick="sendMsg()" class="w-9 h-9 bg-blue-500 hover:bg-blue-600 text-white rounded-full flex items-center justify-center transition-colors flex-shrink-0">➤</button>
            </div>
        </div>
    </div>
</body>
</html>
'''

# ================= 装饰器 =================

def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            if request.path.startswith('/api/'):
                return jsonify({'success': False, 'error': '未登录', 'redirect': '/login'}), 401
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            if request.path.startswith('/api/'):
                return jsonify({'success': False, 'error': '未登录'}), 401
            return redirect('/login')
        if not session.get('is_admin'):
            return redirect('/')
        return f(*args, **kwargs)
    return decorated

# ================= 路由 =================

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect('/login')
    username = session.get('username', '用户')
    # 纯字符串替换，不走 Jinja2 渲染引擎，避免各种兼容问题
    html = MAIN_TEMPLATE.replace('{{username}}', username)
    return html, 200, {'Content-Type': 'text/html; charset=utf-8'}

@app.route('/admin')
@admin_required
def admin_page():
    return render_template_string(ADMIN_TEMPLATE)

@app.route('/login')
def login_page():
    if 'user_id' in session:
        return redirect('/admin' if session.get('is_admin') else '/')
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''
    is_admin = data.get('is_admin', 0)
    if not username or not password:
        return jsonify({'success': False, 'error': '用户名和密码不能为空'})
    if not __import__('re').match(r'^[a-zA-Z0-9_]{4,20}$', username):
        return jsonify({'success': False, 'error': '用户名格式不正确'})
    if len(password) < 6:
        return jsonify({'success': False, 'error': '密码至少6位'})
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT id FROM users WHERE username=%s', (username,))
            if cur.fetchone():
                return jsonify({'success': False, 'error': '用户名已存在'})
            cur.execute('INSERT INTO users (username, password, is_admin) VALUES (%s, %s, %s)',
                        (username, hash_password(password), is_admin))
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        conn.close()

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''
    if not username or not password:
        return jsonify({'success': False, 'error': '请输入用户名和密码'})
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT id, username, password, is_admin FROM users WHERE username=%s', (username,))
            user = cur.fetchone()
        if not user or user['password'] != hash_password(password):
            return jsonify({'success': False, 'error': '用户名或密码错误'})
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['is_admin'] = bool(user['is_admin'])
        return jsonify({'success': True, 'is_admin': bool(user['is_admin'])})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        conn.close()

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'success': True})

# ---- 管理后台 API ----

@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def admin_stats():
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT COUNT(*) as cnt FROM users')
            total_users = cur.fetchone()['cnt']
            cur.execute('SELECT COUNT(*) as cnt FROM users WHERE is_admin=1')
            admin_count = cur.fetchone()['cnt']
            cur.execute('SELECT COUNT(*) as cnt FROM detection_logs')
            total_detections = cur.fetchone()['cnt']
            cur.execute("SELECT COUNT(*) as cnt FROM detection_logs WHERE DATE(created_at)=CURDATE()")
            today_detections = cur.fetchone()['cnt']
        conn.commit()
        return jsonify({'success': True, 'stats': {
            'total_users': total_users,
            'admin_count': admin_count,
            'total_detections': total_detections,
            'today_detections': today_detections
        }})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        conn.close()

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def admin_users():
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute('''
                SELECT u.id, u.username, u.is_admin, u.created_at,
                       COUNT(d.id) as detection_count
                FROM users u
                LEFT JOIN detection_logs d ON u.id = d.user_id
                GROUP BY u.id ORDER BY u.created_at DESC
            ''')
            users = cur.fetchall()
        conn.commit()
        for u in users:
            u['created_at'] = str(u['created_at'])[:19]
        return jsonify({'success': True, 'users': users})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        conn.close()

@app.route('/api/admin/delete_user', methods=['POST'])
@admin_required
def admin_delete_user():
    data = request.get_json()
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': '缺少用户ID'})
    if user_id == session['user_id']:
        return jsonify({'success': False, 'error': '不能删除自己'})
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute('DELETE FROM detection_logs WHERE user_id=%s', (user_id,))
            cur.execute('DELETE FROM users WHERE id=%s AND is_admin=0', (user_id,))
        conn.commit()
        return jsonify({'success': True, 'message': '用户已删除'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        conn.close()

@app.route('/api/admin/logs', methods=['GET'])
@admin_required
def admin_logs():
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute('''
                SELECT d.id, u.username, d.model_name, d.image_name,
                       d.detections, d.created_at
                FROM detection_logs d
                LEFT JOIN users u ON d.user_id = u.id
                ORDER BY d.created_at DESC LIMIT 200
            ''')
            logs = cur.fetchall()
        conn.commit()
        for l in logs:
            l['created_at'] = str(l['created_at'])[:19]
        return jsonify({'success': True, 'logs': logs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        conn.close()

@app.route('/api/admin/change_password', methods=['POST'])
@admin_required
def admin_change_password():
    data = request.get_json()
    new_password = (data.get('new_password') or '').strip()
    if new_password and len(new_password) < 6:
        return jsonify({'success': False, 'error': '密码至少6位'})
    if not new_password:
        return jsonify({'success': False, 'error': '请输入新密码'})
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute('UPDATE users SET password=%s WHERE id=%s', (hash_password(new_password), session['user_id']))
        conn.commit()
        return jsonify({'success': True, 'message': '密码修改成功'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        conn.close()

# ---- 目标检测相关 ----

@app.route('/api/load_model', methods=['POST'])
@login_required
def load_model():
    try:
        if 'model' not in request.files:
            return jsonify({'success': False, 'error': '请求中未包含模型文件'}), 400
        file = request.files['model']
        if file.filename == '':
            return jsonify({'success': False, 'error': '未选择文件'}), 400
        if not allowed_file(file.filename, ALLOWED_MODEL_EXTENSIONS):
            return jsonify({'success': False, 'error': '请上传 .pt 或 .pth 文件'}), 400

        model_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{model_id}_{filename}")
        file.save(model_path)

        try:
            original_torch_load = torch.load
            @functools.wraps(original_torch_load)
            def safe_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            torch.load = safe_load
            try:
                model = YOLO(model_path)
            finally:
                torch.load = original_torch_load

            model_cache[model_id] = {'model': model, 'path': model_path, 'name': filename}
            return jsonify({'success': True, 'model_id': model_id, 'model_name': filename, 'classes': model.names})
        except Exception as e:
            if os.path.exists(model_path):
                os.remove(model_path)
            return jsonify({'success': False, 'error': f'模型加载失败: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/detect', methods=['POST'])
@login_required
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': '未上传图片'}), 400
        model_id = request.form.get('model_id')
        if not model_id or model_id not in model_cache:
            return jsonify({'success': False, 'error': '模型未加载或已失效'}), 400

        image_file = request.files['image']
        if not allowed_file(image_file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({'success': False, 'error': '不支持的图片格式'}), 400

        conf = float(request.form.get('conf_threshold', 0.25))
        iou = float(request.form.get('iou_threshold', 0.45))

        img_id = str(uuid.uuid4())
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{img_id}_{secure_filename(image_file.filename)}")
        image_file.save(img_path)

        try:
            img = cv2.imread(img_path)
            if img is None:
                return jsonify({'success': False, 'error': '无法读取图片'}), 400

            model = model_cache[model_id]['model']
            results = model(img, conf=conf, iou=iou)
            result_obj = results[0]

            plotted = cv2.cvtColor(result_obj.plot(), cv2.COLOR_BGR2RGB)
            buf = BytesIO()
            Image.fromarray(plotted).save(buf, format='JPEG', quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            detections, class_counts = [], {}
            detection_count = 0
            if result_obj.boxes is not None:
                detection_count = len(result_obj.boxes)
                for box in result_obj.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf_val = float(box.conf[0])
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    detections.append({'class_id': cls_id, 'class_name': cls_name,
                                       'confidence': round(conf_val, 2),
                                       'bbox': [float(x) for x in box.xyxy[0].tolist()]})

            # 写入检测日志
            conn = get_db()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        'INSERT INTO detection_logs (user_id, model_name, image_name, detections, conf_threshold, iou_threshold) VALUES (%s, %s, %s, %s, %s, %s)',
                        (session['user_id'], model_cache[model_id]['name'], image_file.filename, detection_count, conf, iou)
                    )
                conn.commit()
            except Exception as log_err:
                print(f"日志写入失败: {log_err}")
            finally:
                conn.close()

            return jsonify({'success': True, 'total_detections': len(detections),
                            'class_counts': class_counts,
                            'image': f'data:image/jpeg;base64,{img_b64}',
                            'detections': detections})
        finally:
            if os.path.exists(img_path):
                os.remove(img_path)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/unload_model', methods=['POST'])
@login_required
def unload_model():
    try:
        data = request.get_json()
        model_id = data.get('model_id') if data else None
        if model_id and model_id in model_cache:
            p = model_cache[model_id]['path']
            if os.path.exists(p):
                os.remove(p)
            del model_cache[model_id]
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ================= AI 助手聊天 =================

QA_KNOWLEDGE = [
    # 上传与加载模型
    ("怎么上传模型|上传模型|加载模型|上传.pt|.pt文件", 
     "📦 <b>上传模型步骤：</b><br>1. 点击模型上传区域或直接拖拽文件<br>2. 选择您的 <code>.pt</code> 或 <code>.pth</code> 模型文件<br>3. 等待模型加载完成（大型模型可能需要几分钟）<br>4. 加载成功后会自动显示识别到的类别数量<br><br>💡 <b>提示：</b>首次使用建议使用 YOLOv8 官方的 <code>yolov8n.pt</code> 测试"),
    ("模型加载失败|模型加载不了|加载模型出错", 
     "❌ <b>模型加载失败常见原因：</b><br>1. <b>PyTorch 版本不匹配</b> — 训练模型用的 PyTorch 版本与当前不同<br>2. <b>文件损坏</b> — 重新下载或导出模型<br>3. <b>格式不对</b> — 确保是 <code>.pt</code> 或 <code>.pth</code>，不是 <code>.onnx</code><br>4. <b>磁盘空间不足</b><br><br>🔧 <b>建议：</b>用 <code>yolo export model=best.pt format=onnx</code> 转换为 ONNX 格式尝试"),
    # 检测相关
    ("检测不到|检测不到目标|没有检测结果|检测失败", 
     "🔍 <b>检测不到目标？试试以下方法：</b><br>1. <b>降低置信度阈值</b> — 默认 0.25，可调到 0.1 或更低<br>2. <b>检查模型类别</b> — 确保模型识别的类别与图片内容匹配<br>3. <b>图片质量</b> — 确保图片清晰、目标可见<br>4. <b>图片尺寸</b> — 过大或过小的图片可能影响检测效果<br>5. <b>IoU 阈值过高</b> — 适当降低 IoU 阈值"),
    ("置信度|conf|confidence|阈值怎么调", 
     "⚙️ <b>置信度阈值说明：</b><br>• 范围：0.05 ~ 0.95<br>• <b>值越低</b>：检测越灵敏，会出现更多框（可能有误检）<br>• <b>值越高</b>：检测越严格，框越少（更精准）<br><br>💡 <b>建议：</b><br>• 通用检测：0.25（默认）<br>• 追求精度：0.5~0.7<br>• 宽松检测：0.1~0.2"),
    ("iou|IoU|NMS|非极大值抑制", 
     "📐 <b>IoU 阈值（非极大值抑制）：</b><br>• 范围：0.05 ~ 0.95<br>• 用于去除重叠的检测框<br>• <b>值越低</b>：去重叠越严格，框越少<br>• <b>值越高</b>：去重叠越宽松，允许更多重叠框<br><br>💡 <b>建议：</b>默认 0.45 适合大多数场景"),
    ("处理时间|速度|很慢|多久", 
     "⏱️ <b>处理时间说明：</b><br>• 取决于：图片尺寸、模型大小、GPU/CPU<br>• GPU 加速时：通常 50-500ms<br>• CPU 运行时：通常 1-10秒<br><br>🚀 <b>加速建议：</b><br>1. 使用 NVIDIA 显卡（自动启用 CUDA）<br>2. 缩小图片尺寸<br>3. 使用更小的模型（如 yolov8n）"),
    # 模型训练相关
    ("怎么训练|训练模型|yolo train", 
     "🧠 <b>YOLOv8 模型训练基础：</b><br><code>yolo train data=coco128.yaml model=yolov8n.pt epochs=100</code><br><br>📋 <b>主要参数：</b><br>• <code>data</code> — 数据集配置文件路径<br>• <code>model</code> — 初始模型（n/s/m/l/x 越大越准越慢）<br>• <code>epochs</code> — 训练轮数（越多越慢但可能更好）<br>• <code>imgsz</code> — 输入图片尺寸（默认640）<br>• <code>batch</code> — 批次大小（根据显存调整）"),
    ("怎么导出|导出模型|export", 
     "📤 <b>模型导出命令：</b><br><code>yolo export model=best.pt format=onnx</code><br><br>📋 <b>支持的格式：</b><br>• <code>onnx</code> — 通用格式，跨平台<br>• <code>torchscript</code> — PyTorch 专用<br>• <code>openvino</code> — Intel CPU 优化<br>• <code>tensorflow saved_model</code> — TensorFlow 用<br>• <code>coreml</code> — Apple 设备"),
    # CUDA / GPU
    ("cuda|gpu|显卡|英伟达", 
     "🖥️ <b>GPU 加速说明：</b><br>• 有 NVIDIA 显卡且装了 CUDA → 自动使用 GPU<br>• 无显卡或无 CUDA → 自动回退到 CPU<br><br>🔧 <b>检查 CUDA 是否可用：</b>服务启动时会显示在状态栏<br><br>💡 <b>没有 GPU？</b>可以使用 Google Colab（免费 GPU）、 Kaggle Notebook 或在线推理服务"),
    # 系统使用
    ("怎么用|使用说明|教程|help|帮助", 
     "👋 <b>欢迎使用 YOLOv8 目标检测系统！</b><br><br>📋 <b>使用步骤：</b><br>1️⃣ <b>上传模型</b> — 上传训练好的 <code>.pt</code> 文件<br>2️⃣ <b>上传图片</b> — 支持 JPG/PNG/WebP<br>3️⃣ <b>调整参数</b> — 置信度和 IoU 阈值<br>4️⃣ <b>开始检测</b> — 点击按钮即可<br>5️⃣ <b>下载结果</b> — 保存带检测框的图片<br><br>💬 有任何问题都可以问我！"),
]

def get_answer(question):
    q = question.lower().strip()
    best_score = 0
    best_answer = None
    for keywords, answer in QA_KNOWLEDGE:
        for kw in keywords.split('|'):
            if kw in q:
                score = len(kw)
                if score > best_score:
                    best_score = score
                    best_answer = answer
    if best_answer:
        return best_answer
    return ("🤔 <b>这个问题我还在学习中～</b><br><br>"
            "目前我能解答的问题包括：<br>"
            "• 模型上传与加载<br>"
            "• 置信度 / IoU 阈值设置<br>"
            "• 检测不到目标的排查<br>"
            "• YOLOv8 训练与导出命令<br>"
            "• GPU / CUDA 加速说明<br>"
            "• 系统使用帮助<br><br>"
            "💡 <b>建议：</b>尝试用关键词提问，或查看页面上的快捷问题按钮")

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    data = request.get_json()
    message = (data.get('message') or '').strip()
    if not message:
        return jsonify({'reply': '请输入问题'})
    reply = get_answer(message)
    return jsonify({'reply': reply, 'question': message})

# ================= 健康检查 =================

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'running', 'torch_version': torch.__version__,
                    'cuda_available': torch.cuda.is_available(),
                    'loaded_models': len(model_cache)})

# ================= 启动入口 =================

def open_browser():
    import time
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    print("=" * 60)
    print("        YOLOv8 目标检测系统  (含管理后台)")
    print("=" * 60)
    print(f"PyTorch 版本 : {torch.__version__}")
    print(f"CUDA 可用    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备    : {torch.cuda.get_device_name(0)}")
    print("数据库初始化在后台运行中...")
    threading.Thread(target=init_db, daemon=True).start()
    print("=" * 60)
    print("▶  访问地址: http://127.0.0.1:5000")
    print("▶  管理后台: http://127.0.0.1:5000/admin")
    print("▶  默认管理员: admin / admin123")
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
