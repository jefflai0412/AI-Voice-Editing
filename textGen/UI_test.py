import gradio as gr
import text_merge_1 as step1
from pydub import AudioSegment
import copy
import os
from functools import partial
import sys

sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from audioGen import xtts_syn_api


# 預設的空 transcription_list
transcription_list = []
original_list = []
modified_entries = []  # 用於保存被修改的條目


def split_audio(input_audio,output_folder,list):
    audio = AudioSegment.from_file(input_audio)
    os.makedirs(output_folder, exist_ok=True)
    # 按時間戳分割音頻
    for i, entry in enumerate(list):
        start_time = int(entry["start"] * 1000)  # 開始時間 (毫秒)
        end_time = int(entry["end"] * 1000)      # 結束時間 (毫秒)
        split_audio = audio[start_time:end_time]  # 分割音頻片段

        # 儲存分割的音頻到指定資料夾
        output_path = os.path.join(output_folder, f"segment_{i + 1}.wav")
        split_audio.export(output_path, format="wav")

# 第一頁的函數（處理訓練音檔）
def process_training_audio(input_audio):
    transcription_list = step1.text_merge_1(input_audio, "tiny", "textGen/UI/output.txt", 2.0)
    # 創建分割音頻的輸出資料夾
    output_folder= "textGen/UI/split_audio"
    split_audio(input_audio,output_folder,transcription_list)
    return f"訓練音檔處理完成，分割音頻已保存至：{output_folder}"

# 第二頁的函數（處理編輯音檔）
def process_audio(input_audio):
    global original_list
    original_list = step1.text_merge_1(input_audio,"tiny","textGen/UI/output.txt", 2.0)
    for entry in original_list:
        entry["status"] = ""

    #print("original_list",original_list)
    #process_list = copy.deepcopy(original_list)
    updated_list = "\n".join(
        [
            f"[{entry['start']:.2f}s - {entry['end']:.2f}s]: {entry['text']}"
            for entry in original_list
        ]
    )
    next_index, next_text = load_entry(0)
    return updated_list,next_index, next_text

def load_entry(index):
    global current_index  # 使用全局變數記錄當前索引
    global original_list
    try:
        index = int(index)
        if 0 <= index < len(original_list):  
            current_index = index  
            entry = original_list[index]
            return str(index), entry["text"]
        else:  
            entry = original_list[current_index]
            return str(current_index), entry["text"]
    except (ValueError, IndexError):  
        entry = original_list[current_index]
        return str(current_index), entry["text"]

def update_entry(index, new_text_update):
    global original_list, current_index, modified_entries
    try:
        index = int(index)
        if 0 <= index < len(original_list):
            # 更新時間戳和文字（只有在文字改變時才更新）
            if new_text_update.strip() and new_text_update.strip() != original_list[index]["text"]:
                original_list[index]["text"] = new_text_update.strip()
                if original_list[index]["status"] != "new":
                    original_list[index]["status"] = "modified"
                modified_entries.append({
                    "index": index,
                    "original_text": original_list[index]["text"],
                    "new_text":new_text_update.strip()
                })

            # 自動跳到下一條（若非最後一條）
            if index < len(original_list) - 1:
                current_index = index + 1
            else:
                current_index = index
        else:
            next_index, next_text = load_entry(str(index))
            return "索引超出範圍", next_index, next_text

        updated_list = "\n".join(
            [
                f"[{entry['start']:.2f}s - {entry['end']:.2f}s]: {entry['text']}"
                for entry in original_list
            ]
        )
        next_index, next_text = load_entry(current_index)
        return updated_list, next_index, next_text
    except ValueError:
        return "Error: 請輸入正確的數值格式！", str(index), new_text_update
    except Exception as e:
        return f"Error: {str(e)}", str(index), new_text_update
    
def add_entry(direction):
    global original_list
    global current_index

    # 構造新資料
    new_entry = {
        "start": 0.0,
        "end": 0.0,
        "text": "",
        "status": "new"
    }

    if direction == "up":  # 插入到當前索引的上方
        if 0 <= current_index < len(original_list):
            original_list.insert(current_index, new_entry)
        else:
            original_list.insert(0, new_entry)  # 如果索引無效，插入到最前面
    elif direction == "down":  # 插入到當前索引的下方
        if 0 <= current_index < len(original_list):
            original_list.insert(current_index + 1, new_entry)
            current_index += 1  # 更新索引到新條目位置
        else:
            original_list.append(new_entry)  # 如果索引無效，插入到最後面

    # 構造輸出的顯示列表
    updated_list = "\n".join(
        [
            f"[{entry['start']:.2f}s - {entry['end']:.2f}s]: {entry['text']}"
            for entry in original_list
        ]
    )

    return updated_list, current_index, new_entry["text"]

def generate_audio(input_audio_path, output_path="audioGen/output"):
    global original_list
    # 遍歷原始列表，生成有變化的語音
    os.makedirs(output_path, exist_ok=True)
    # 初始化一個空白音檔
    input_audio = AudioSegment.from_file(input_audio_path)
    final_audio = AudioSegment.silent(duration=0)
    # preload model
    model = xtts_syn_api.xtts_model()

    for index,entry in enumerate(original_list):
        if entry["status"] in ("modified", "new"):
            filename = f"{index}.wav"
            file_path = os.path.join(output_path, filename)
            model.synthesis(entry["text"], "audioGen/reference/speaker/speaker_bryan.wav", file_path)
            segment = AudioSegment.from_file(file_path)
        else:
            # 從基礎音檔中提取對應的時間片段
            start_ms = int(entry["start"] * 1000)  # 開始時間 (毫秒)
            end_ms = int(entry["end"] * 1000)      # 結束時間 (毫秒)
            segment = input_audio[start_ms:end_ms]
        final_audio += segment

    save_path = os.path.join(output_path, "final.mp4")
    final_audio.export(save_path, format="mp4")

    updated_list = "\n".join(
        [
            f"[{entry['start']:.2f}s - {entry['end']:.2f}s]: {entry['text']}"
            for entry in original_list
        ]
    )
    return updated_list

# Gradio 界面設計
with gr.Blocks() as demo:
    with gr.Tabs():
        # 第一頁：訓練音檔
        with gr.Tab("訓練音檔"):
            gr.Markdown("### 上傳訓練音檔")
            training_audio_input = gr.Audio(type="filepath", label="上傳訓練音檔")
            process_training_button = gr.Button("處理訓練音檔")
            training_result = gr.Textbox(label="處理結果")
            process_training_button.click(
                process_training_audio,
                inputs=[training_audio_input],
                outputs=[training_result],
            )

        # 第二頁：更改時間戳
        with gr.Tab("更改時間戳"):
            gr.Markdown("### 上傳音檔以更改時間戳")
            with gr.Row():
                audio_input = gr.Audio(type="filepath", label="上傳音檔")
                process_button = gr.Button("處理音檔")

            transcription_display = gr.Textbox(
                value="請上傳音頻以顯示轉錄內容",
                lines=10,
                label="目前的內容",
            )
    
            # 編輯部分
            with gr.Row():
                with gr.Column(scale=1):
                    index_input = gr.Number(label="條目索引 (從 0 開始)", value=0)
                with gr.Column(scale=3):
                    text_input = gr.Textbox(label="新的文字 (可選)", placeholder="輸入新的文字")

            update_button = gr.Button("更新內容")
            addup_button = gr.Button("向上插入內容")
            adddown_button = gr.Button("向下插入內容")

            change_display = gr.Textbox(
                value="按下開始生成才會更新",
                lines=10,
                label="生成的內容",
            )
            finish_button = gr.Button("開始生成")

            # 綁定功能
            process_button.click(
                process_audio,
                inputs=[audio_input],
                outputs=[transcription_display,index_input,text_input],
            )
            index_input.change(
                lambda idx: load_entry(idx),
                inputs=[index_input],
                outputs=[index_input, text_input],
            )
            update_button.click(
                update_entry,
                inputs=[index_input,  text_input],
                outputs=[
                    transcription_display,
                    index_input,
                    text_input,
                ],
            )
            addup_button.click(
                fn=partial(add_entry, direction="up"),
                inputs = [],
                outputs=[
                    transcription_display,
                    index_input,
                    text_input,
                ],
            )
            adddown_button.click(
                fn=partial(add_entry, direction="down"),
                inputs = [],
                outputs=[
                    transcription_display,
                    index_input,
                    text_input,
                ],
            )
            finish_button.click(
                generate_audio,
                inputs=[audio_input],
                outputs=[
                    change_display
                ]
            )

# 啟動 Gradio
demo.launch()
