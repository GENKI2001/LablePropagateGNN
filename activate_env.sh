#!/bin/bash

# GSLCodes プロジェクト用仮想環境activateスクリプト
# 使用方法: source activate_env.sh

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 仮想環境のパス
VENV_PATH="$SCRIPT_DIR/env"

echo "=== GSLCodes 仮想環境activateスクリプト ==="

# 仮想環境が存在するかチェック
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ 仮想環境が見つかりません: $VENV_PATH"
    echo "仮想環境を作成しますか？ (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "仮想環境を作成中..."
        python3 -m venv "$VENV_PATH"
        echo "✅ 仮想環境を作成しました"
    else
        echo "仮想環境の作成をキャンセルしました"
        exit 1
    fi
fi

# 仮想環境をactivate
echo "🔧 仮想環境をactivate中..."
source "$VENV_PATH/bin/activate"

# 仮想環境が正しくactivateされたかチェック
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✅ 仮想環境がactivateされました: $VIRTUAL_ENV"
    echo "🐍 Python: $(which python)"
    echo "📦 pip: $(which pip)"
    
    # 必要なパッケージがインストールされているかチェック
    echo "📋 必要なパッケージをチェック中..."
    if ! python -c "import torch" 2>/dev/null; then
        echo "⚠️  PyTorchがインストールされていません"
        echo "requirements.txtからパッケージをインストールしますか？ (y/n)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            pip install -r requirements.txt
            echo "✅ パッケージのインストールが完了しました"
        fi
    else
        echo "✅ 必要なパッケージは既にインストールされています"
    fi
    
    echo ""
    echo "🎉 準備完了！以下のコマンドが使用できます："
    echo "  python index.py                    # メイン実験"
    echo "  python run_label_analysis.py       # ラベル相関分析"
    echo "  python custom_dataset_creator.py   # カスタムデータセット作成"
    echo ""
    echo "仮想環境を終了するには 'deactivate' を実行してください"
    
else
    echo "❌ 仮想環境のactivateに失敗しました"
    exit 1
fi 