@echo off
REM GSLCodes プロジェクト用仮想環境activateスクリプト (Windows)

echo === GSLCodes 仮想環境activateスクリプト ===

REM スクリプトのディレクトリを取得
set SCRIPT_DIR=%~dp0
set VENV_PATH=%SCRIPT_DIR%env

REM 仮想環境が存在するかチェック
if not exist "%VENV_PATH%" (
    echo ❌ 仮想環境が見つかりません: %VENV_PATH%
    echo 仮想環境を作成しますか？ (y/n)
    set /p response=
    if /i "%response%"=="y" (
        echo 仮想環境を作成中...
        python -m venv "%VENV_PATH%"
        echo ✅ 仮想環境を作成しました
    ) else (
        echo 仮想環境の作成をキャンセルしました
        pause
        exit /b 1
    )
)

REM 仮想環境をactivate
echo 🔧 仮想環境をactivate中...
call "%VENV_PATH%\Scripts\activate.bat"

REM 仮想環境が正しくactivateされたかチェック
if defined VIRTUAL_ENV (
    echo ✅ 仮想環境がactivateされました: %VIRTUAL_ENV%
    echo 🐍 Python: %VIRTUAL_ENV%\Scripts\python.exe
    echo 📦 pip: %VIRTUAL_ENV%\Scripts\pip.exe
    
    REM 必要なパッケージがインストールされているかチェック
    echo 📋 必要なパッケージをチェック中...
    python -c "import torch" 2>nul
    if errorlevel 1 (
        echo ⚠️  PyTorchがインストールされていません
        echo requirements.txtからパッケージをインストールしますか？ (y/n)
        set /p response=
        if /i "%response%"=="y" (
            pip install -r requirements.txt
            echo ✅ パッケージのインストールが完了しました
        )
    ) else (
        echo ✅ 必要なパッケージは既にインストールされています
    )
    
    echo.
    echo 🎉 準備完了！以下のコマンドが使用できます：
    echo   python index.py                    # メイン実験
    echo   python run_label_analysis.py       # ラベル相関分析
    echo   python custom_dataset_creator.py   # カスタムデータセット作成
    echo.
    echo 仮想環境を終了するには 'deactivate' を実行してください
    
) else (
    echo ❌ 仮想環境のactivateに失敗しました
    pause
    exit /b 1
)

pause 