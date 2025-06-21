#!/bin/bash

# GSLCodes カスタムグラフデータセット作成スクリプト

# デフォルトパラメータ
DEFAULT_NODES=1200
DEFAULT_AVG_DEGREE=3.0
DEFAULT_NAME="CustomGraph"
DEFAULT_PATTERN="default"
CUSTOM_PATTERN_FILE=""
CUSTOM_PATTERN_STRING=""

# ヘルプメッセージ
show_help() {
    echo "使用方法: $0 [オプション]"
    echo ""
    echo "オプション:"
    echo "  -n, --nodes NUM        ノード数 (デフォルト: $DEFAULT_NODES)"
    echo "  -d, --degree NUM       目標平均次数 (デフォルト: $DEFAULT_AVG_DEGREE)"
    echo "  -p, --pattern PATTERN  接続パターン (デフォルト: $DEFAULT_PATTERN)"
    echo "                         - default: デフォルトパターン"
    echo "                         - chain: チェーンパターン"
    echo "                         - full: 完全グラフパターン"
    echo "                         - star: スター型パターン"
    echo "  -f, --file FILE        カスタム接続パターンファイルを読み込み"
    echo "  -s, --string STRING    カスタム接続パターンを文字列で指定"
    echo "                         例: '0:1,2;1:0,3;2:0,4;3:1,4;4:2,3'"
    echo "  -c, --clear            キャッシュを削除してから実行"
    echo "  -h, --help             このヘルプを表示"
    echo ""
    echo "接続パターンファイル形式 (JSON):"
    echo '  {'
    echo '    "0": [2, 3, 4],'
    echo '    "1": [3, 4],'
    echo '    "2": [0],'
    echo '    "3": [1, 4],'
    echo '    "4": [0, 1, 3]'
    echo '  }'
    echo ""
    echo "例:"
    echo "  $0                                    # デフォルト設定で実行"
    echo "  $0 -n 2000 -d 4.0                     # ノード数2000、平均次数4.0"
    echo "  $0 -p chain -n 1500                   # チェーンパターン、ノード数1500"
    echo "  $0 -f pattern.json                    # カスタムパターンファイルを読み込み"
    echo "  $0 -s '0:1,2;1:0,3;2:0,4;3:1,4;4:2,3' # カスタムパターンを文字列で指定"
    echo "  $0 -c -p full -n 1000 -d 5.0          # キャッシュ削除、完全グラフ、ノード数1000、平均次数5.0"
}

# パターン文字列をPython辞書に変換
parse_pattern_string() {
    local pattern_str="$1"
    local python_dict="{"
    local first=true
    
    IFS=';' read -ra patterns <<< "$pattern_str"
    for pattern in "${patterns[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            python_dict+=","
        fi
        
        IFS=':' read -ra parts <<< "$pattern"
        local source="${parts[0]}"
        local targets="${parts[1]}"
        
        python_dict+="$source: ["
        IFS=',' read -ra target_list <<< "$targets"
        local target_first=true
        for target in "${target_list[@]}"; do
            if [ "$target_first" = true ]; then
                target_first=false
            else
                python_dict+=","
            fi
            python_dict+="$target"
        done
        python_dict+="]"
    done
    python_dict+="}"
    echo "$python_dict"
}

# パラメータ解析
NODES=$DEFAULT_NODES
AVG_DEGREE=$DEFAULT_AVG_DEGREE
NAME=$DEFAULT_NAME
PATTERN=$DEFAULT_PATTERN
CLEAR_CACHE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--nodes)
            NODES="$2"
            shift 2
            ;;
        -d|--degree)
            AVG_DEGREE="$2"
            shift 2
            ;;
        -p|--pattern)
            PATTERN="$2"
            shift 2
            ;;
        -f|--file)
            CUSTOM_PATTERN_FILE="$2"
            shift 2
            ;;
        -s|--string)
            CUSTOM_PATTERN_STRING="$2"
            shift 2
            ;;
        -c|--clear)
            CLEAR_CACHE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "不明なオプション: $1"
            show_help
            exit 1
            ;;
    esac
done

# キャッシュ削除
if [ "$CLEAR_CACHE" = true ]; then
    echo "🗂️  キャッシュを削除中..."
    rm -rf /tmp/CustomGraph
    echo "✅ キャッシュを削除しました"
fi

# 接続パターンの設定
if [ -n "$CUSTOM_PATTERN_FILE" ]; then
    # カスタムパターンファイルを読み込み
    echo "📊 カスタム接続パターンファイルを読み込み: $CUSTOM_PATTERN_FILE"
    if [ ! -f "$CUSTOM_PATTERN_FILE" ]; then
        echo "❌ ファイルが見つかりません: $CUSTOM_PATTERN_FILE"
        exit 1
    fi
    python3 -c "
import utils.custom_dataset_creator as cdc
import json

with open('$CUSTOM_PATTERN_FILE', 'r') as f:
    custom_patterns = json.load(f)

# 文字列キーを整数キーに変換
connection_patterns = {int(k): v for k, v in custom_patterns.items()}

dataset = cdc.create_custom_dataset(
    num_nodes=$NODES, 
    name='$NAME', 
    target_avg_degree=$AVG_DEGREE,
    connection_patterns=connection_patterns
)
cdc.analyze_dataset(dataset)
"
elif [ -n "$CUSTOM_PATTERN_STRING" ]; then
    # カスタムパターン文字列を解析
    echo "📊 カスタム接続パターン文字列を解析: $CUSTOM_PATTERN_STRING"
    PATTERN_DICT=$(parse_pattern_string "$CUSTOM_PATTERN_STRING")
    python3 -c "
import utils.custom_dataset_creator as cdc

connection_patterns = $PATTERN_DICT

dataset = cdc.create_custom_dataset(
    num_nodes=$NODES, 
    name='$NAME', 
    target_avg_degree=$AVG_DEGREE,
    connection_patterns=connection_patterns
)
cdc.analyze_dataset(dataset)
"
else
    # プリセットパターンを使用
    case $PATTERN in
        "default")
            echo "📊 デフォルト接続パターンを使用"
            python3 -c "
import utils.custom_dataset_creator as cdc
dataset = cdc.create_custom_dataset(
    num_nodes=$NODES, 
    name='$NAME', 
    target_avg_degree=$AVG_DEGREE
)
cdc.analyze_dataset(dataset)
"
            ;;
        "chain")
            echo "📊 チェーン接続パターンを使用"
            python3 -c "
import utils.custom_dataset_creator as cdc
chain_patterns = {
    0: [1],      # クラス1 → クラス2
    1: [0, 2],   # クラス2 → クラス1, クラス3
    2: [1, 3],   # クラス3 → クラス2, クラス4
    3: [2, 4],   # クラス4 → クラス3, クラス5
    4: [3]       # クラス5 → クラス4
}
dataset = cdc.create_custom_dataset(
    num_nodes=$NODES, 
    name='$NAME', 
    target_avg_degree=$AVG_DEGREE,
    connection_patterns=chain_patterns
)
cdc.analyze_dataset(dataset)
"
            ;;
        "full")
            echo "📊 完全グラフ接続パターンを使用"
            python3 -c "
import utils.custom_dataset_creator as cdc
full_patterns = {
    0: [1, 2, 3, 4],  # クラス1は他のすべてのクラスと接続
    1: [0, 2, 3, 4],  # クラス2は他のすべてのクラスと接続
    2: [0, 1, 3, 4],  # クラス3は他のすべてのクラスと接続
    3: [0, 1, 2, 4],  # クラス4は他のすべてのクラスと接続
    4: [0, 1, 2, 3]   # クラス5は他のすべてのクラスと接続
}
dataset = cdc.create_custom_dataset(
    num_nodes=$NODES, 
    name='$NAME', 
    target_avg_degree=$AVG_DEGREE,
    connection_patterns=full_patterns
)
cdc.analyze_dataset(dataset)
"
            ;;
        "star")
            echo "📊 スター型接続パターンを使用"
            python3 -c "
import utils.custom_dataset_creator as cdc
star_patterns = {
    0: [1, 2, 3, 4],  # クラス1が中心（ハブ）
    1: [0],            # クラス2 → クラス1
    2: [0],            # クラス3 → クラス1
    3: [0],            # クラス4 → クラス1
    4: [0]             # クラス5 → クラス1
}
dataset = cdc.create_custom_dataset(
    num_nodes=$NODES, 
    name='$NAME', 
    target_avg_degree=$AVG_DEGREE,
    connection_patterns=star_patterns
)
cdc.analyze_dataset(dataset)
"
            ;;
        *)
            echo "❌ 不明な接続パターン: $PATTERN"
            echo "利用可能なパターン: default, chain, full, star"
            exit 1
            ;;
    esac
fi

echo ""
echo "🎉 データセット作成完了！"
echo "📁 保存場所: /tmp/CustomGraph"
echo "📊 パラメータ: ノード数=$NODES, 平均次数=$AVG_DEGREE"
if [ -n "$CUSTOM_PATTERN_FILE" ]; then
    echo "📄 パターン: ファイル ($CUSTOM_PATTERN_FILE)"
elif [ -n "$CUSTOM_PATTERN_STRING" ]; then
    echo "📄 パターン: カスタム文字列"
else
    echo "📄 パターン: $PATTERN"
fi