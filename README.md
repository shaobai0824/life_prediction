# 壽命預測保險決策輔助系統

> **作者**: shaobai
> **⚠️ 重要聲明**: 本系統僅供保險業決策輔助使用，非醫療診斷工具。預測結果僅供參考，不代表實際壽命或健康狀況。

## 🎯 專案概述

透過先進的人工智慧技術分析面部特徵，預測個體壽命，主要應用於金融保險業的風險評估。系統採用深度學習模型結合面相識別技術，提供快速、客觀的健康狀況初步評估。

### 🌟 核心特色
- 🤖 **AI 驅動**: PyTorch 深度學習模型
- 🛡️ **金融合規**: 完整的個資保護和監管合規
- 🎨 **中國風界面**: 算命風格的用戶體驗
- ⚡ **快速預測**: 1分鐘內完成分析
- 🔒 **隱私保護**: 即時處理，不保存原始照片

## 🏗️ 技術架構

### 前端技術棧
- **Framework**: React 18 + TypeScript
- **UI Library**: Ant Design (中國風客製化)
- **狀態管理**: Redux Toolkit
- **打包工具**: Vite

### 後端技術棧
- **API Framework**: FastAPI + Python 3.9+
- **AI/ML**: PyTorch + OpenCV
- **資料庫**: SQLite (開發) → PostgreSQL (生產)
- **認證**: JWT + 雙因子認證
- **快取**: Redis

### DevOps & 部署
- **容器化**: Docker + Docker Compose
- **反向代理**: Nginx
- **監控**: Prometheus + Grafana
- **日誌**: ELK Stack

## 🚀 快速開始

### 先決條件
```bash
# Python 3.9+
python --version

# Node.js 18+
node --version

# Docker (可選)
docker --version
```

### 安裝與設定

1. **複製儲存庫**
```bash
git clone https://github.com/your-username/life_prediction.git
cd life_prediction
```

2. **設定後端環境**
```bash
cd src/backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **設定前端環境**
```bash
cd src/frontend
npm install
```

4. **啟動開發服務器**
```bash
# 後端 (Terminal 1)
cd src/backend
uvicorn main:app --reload --port 8000

# 前端 (Terminal 2)
cd src/frontend
npm run dev
```

### 🔧 開發環境設定

#### 環境變數配置
複製 `.env.example` 到 `.env` 並設定必要參數：
```bash
# 資料庫設定
DATABASE_URL=sqlite:///./app.db

# JWT 設定
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256

# AI 模型設定
MODEL_PATH=./models/trained/
```

#### 資料庫初始化
```bash
cd src/backend
python -m alembic upgrade head
```

## 📋 專案結構

```
life_prediction/
├── 📁 data/                    # 資料集管理
│   ├── raw/                   # 原始資料
│   ├── processed/             # 預處理資料
│   └── temp/                  # 暫存資料
├── 📁 models/                 # AI 模型
│   ├── trained/              # 訓練完成的模型
│   ├── checkpoints/          # 訓練檢查點
│   └── metadata/             # 模型元資料
├── 📁 notebooks/             # Jupyter 分析
│   ├── exploratory/         # 資料探索
│   ├── experiments/         # 模型實驗
│   └── reports/             # 分析報告
├── 📁 src/                   # 原始碼
│   ├── backend/             # FastAPI 後端
│   ├── frontend/            # React 前端
│   ├── ml_pipeline/         # ML 資料管道
│   └── common/              # 共用工具
├── 📁 docs/                 # 文件
│   ├── compliance/          # 合規文件
│   ├── security/            # 資安政策
│   └── api/                # API 文檔
├── 📁 tests/               # 測試
├── 📁 .claude/             # Claude Code 協作
└── 📄 CLAUDE.md            # 開發規範
```

## 🧪 測試

### 運行測試
```bash
# 後端測試
cd src/backend
pytest tests/ -v

# 前端測試
cd src/frontend
npm test

# E2E 測試
npm run test:e2e
```

### 測試覆蓋率
```bash
# 後端覆蓋率
pytest --cov=src tests/

# 前端覆蓋率
npm run test:coverage
```

## 🔒 安全性與合規

### 資料保護
- ✅ **AES-256 加密**: 敏感資料靜態加密
- ✅ **TLS 1.3**: 傳輸層安全加密
- ✅ **即時處理**: 照片上傳後立即分析並刪除
- ✅ **審計日誌**: 完整的操作記錄

### 合規檢查
- ✅ **個資法遵循**: 符合台灣個人資料保護法
- ✅ **保險法規**: 決策輔助工具合規要求
- ✅ **AI 倫理**: 公平性和可解釋性
- ✅ **消費者保護**: 透明告知和申訴機制

## 📊 API 文檔

### 主要端點

#### 認證
```
POST /api/v1/auth/login        # 用戶登入
POST /api/v1/auth/register     # 用戶註冊
POST /api/v1/auth/refresh      # 刷新 Token
```

#### 預測
```
POST /api/v1/predict/upload    # 上傳照片進行預測
GET  /api/v1/predict/history   # 查看預測歷史
GET  /api/v1/predict/{id}      # 查看特定預測結果
```

#### 用戶管理
```
GET  /api/v1/users/profile     # 查看用戶資料
PUT  /api/v1/users/profile     # 更新用戶資料
DEL  /api/v1/users/profile     # 刪除用戶帳戶
```

### Swagger 文檔
開發環境啟動後，訪問 `http://localhost:8000/docs` 查看完整 API 文檔。

## 🚀 部署

### Docker 部署
```bash
# 建置映像
docker-compose build

# 啟動服務
docker-compose up -d

# 查看狀態
docker-compose ps
```

### 生產環境部署
1. **環境準備**
```bash
# 設定生產環境變數
export ENVIRONMENT=production
export DATABASE_URL=postgresql://user:pass@host:5432/db
```

2. **資料庫遷移**
```bash
python -m alembic upgrade head
```

3. **啟動服務**
```bash
# 使用 Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

## 🤖 Claude Code 整合

本專案使用 Claude Code 協作開發，請遵循以下原則：

### 開發前檢查
- [ ] ✅ 閱讀 `CLAUDE.md` 中的開發規範
- [ ] 🔍 搜尋現有功能，避免重複開發
- [ ] 📋 使用 TodoWrite 管理複雜任務
- [ ] 🔒 確保符合金融合規要求

### 提交規範
```bash
# 使用 Conventional Commits
git commit -m "feat(auth): 新增雙因子認證功能"
git commit -m "fix(api): 修復預測結果序列化問題"
```

## 🔬 AI 模型

### 模型架構
- **面部檢測**: MTCNN
- **特徵提取**: ResNet-50 backbone
- **壽命預測**: 自定義回歸頭
- **準確度目標**: 80%+

### 訓練資料
- **公開資料集**: CelebA、IMDB-WIKI
- **合成資料**: GAN 生成的面相資料
- **統計資料**: 保險業壽命統計

### 模型更新
```bash
# 重新訓練模型
python src/ml_pipeline/train_model.py

# 評估模型性能
python src/ml_pipeline/evaluate_model.py

# 部署新模型
python src/ml_pipeline/deploy_model.py
```

## 📈 監控與運維

### 健康檢查
```bash
# API 健康檢查
curl http://localhost:8000/health

# 資料庫連接檢查
curl http://localhost:8000/health/db

# AI 模型檢查
curl http://localhost:8000/health/model
```

### 監控指標
- **API 響應時間**: < 1000ms
- **預測準確度**: > 80%
- **系統可用性**: > 99.9%
- **錯誤率**: < 0.1%

## 🤝 貢獻指南

### 開發流程
1. Fork 本儲存庫
2. 建立功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交變更 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 開啟 Pull Request

### 程式碼規範
- **Python**: 遵循 PEP 8，使用 Black 格式化
- **TypeScript**: 遵循 ESLint 規則
- **Git**: 使用 Conventional Commits

## 📞 支援與聯絡

### 技術支援
- **Email**: support@lifepredict.com
- **GitHub Issues**: [專案 Issues 頁面]
- **文檔**: [完整文檔連結]

### 法律與合規
- **隱私權問題**: privacy@lifepredict.com
- **合規諮詢**: compliance@lifepredict.com
- **法律顧問**: legal@lifepredict.com

## 📄 授權條款

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案。

---

**⚠️ 免責聲明**: 本系統僅供保險業決策參考，不構成醫療建議。實際保險決策應由專業人員綜合評估後決定。

**🔒 隱私保護**: 我們高度重視用戶隱私，詳細的隱私政策請參見 [Privacy Policy](docs/compliance/privacy_policy_template.md)。