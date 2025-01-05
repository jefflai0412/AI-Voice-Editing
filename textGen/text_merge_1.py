import whisper

def audio2text_to_list(inputpath, model_name): #選擇模型並回傳文字與時間戳
    # 加載 Whisper 模型
    model = whisper.load_model(model_name)
    # 轉錄音頻
    result = model.transcribe(inputpath, task="transcribe", word_timestamps=True)
    # 將結果存儲到 list
    transcription_list = [
        {"start": segment["start"], "end": segment["end"], "text": segment["text"]}
        for segment in result["segments"]
    ]
    return transcription_list

def merge_timestamp_list(transcription_list, merge_threshold): #確保訓練集的資料長度
    merged_entries = []
    current_text = ""
    current_start = None
    current_end = None

    # 遍歷時間戳與文字內容
    for segment in transcription_list:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]

        # 如果當前段落太短，合併文字
        if current_start is None:
            current_start = start
            current_end = end
            current_text = text.strip()
        elif (end - current_start) <= merge_threshold:
            current_end = end
            current_text =current_text.strip()+ "," + text.strip()
        else:
            # 存儲已合併的段落
            merged_entries.append(
                {"start": current_start, "end": current_end, "text": current_text + "。"}
            )
            # 重置合併狀態
            current_start = start
            current_end = end
            current_text = text.strip()

    # 存儲最後一個段落
    if current_text:
        merged_entries.append(
            {"start": current_start, "end": current_end, "text": current_text + "。"}
        )

    return merged_entries

def save_merged_to_txt(merged_list, output_path): #將文字與時間輟存成txt顯示
    # 將合併結果寫入文字檔案
    with open(output_path, "w", encoding="utf-8") as file:
        for entry in merged_list:
            file.write(f"[{entry['start']:.2f}s - {entry['end']:.2f}s]: {entry['text']}\n")
    print(f"Saved merged transcription to: {output_path}")

def adjust_timestamps(merged_list): #微調每段的時間
    updated_timestamps = []

    for i in range(len(merged_list)):
        entry = merged_list[i]
        start = entry["start"]
        end = entry["end"]
        text = entry["text"]

        # 結束時間延長 0.1 秒
        new_end = round(end + 0.1, 2)

        # 如果不是最後一段，讓下一段的開始時間同步更新
        if i < len(merged_list) - 1:
            merged_list[i + 1]["start"] = new_end

        # 更新當前段的時間戳
        updated_timestamps.append(
            {"start": start, "end": new_end, "text": text}
        )
    return updated_timestamps

def text_merge_1(input,model,ouput,merge_threshold):
    # 步驟 1: 音頻轉錄為 list
    transcription_list = audio2text_to_list(input, model)
    # 步驟 2: 合併時間戳
    #merged_transcription_list = merge_timestamp_list(transcription_list, merge_threshold)
    # 步驟 3: 儲存合併結果為文字檔
    #updated_timestamps=adjust_timestamps(merged_transcription_list)
    
    #save_merged_to_txt(transcription_list, ouput)
    return transcription_list
    

if __name__ == "__main__":
    # 音頻檔案和模型名稱
    input = "textGen/dataset/audio/segment_1.wav"
    model = "tiny"
    merge_threshold = 7.0
    ouput = "textGen/text/test_demo3_merge.txt"
    text_merge_1(input,ouput,model,merge_threshold)
