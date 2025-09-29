製作者:國立屏東大學 唐元宗 許瀚升 賴士欹 謝人傑

影片:
YT:https://youtu.be/Y0mCf9wbXbw

GOOGLE DRIVE:https://drive.google.com/file/d/1Wn3meD_skf0eSZmJQVmDB9F9uKNYn9Ui/view?usp=sharing

Driver Monitor System 
---

Test computer and System:

- Desktop setting: i5-10400, GPU 3080, CUDA 11.3
- System setting: Ubuntu 22.04, Python3.10
- Test Date: 2025/07/31



## Install

GEMINI API_KEY 申請:https://aistudio.google.com/apikey

請將這行
```
註解掉from gemini_chatbot import API_KEY
client = genai.Client(api_key=API_KEY) #換成自己的API KEY
```

```
pip install google-genai
```
"Clone This repo file install "
```
python install -r requirements.txt
```

### TODO

  [o] 新增聊天機器人-基本資訊(即時狀態、狀態查詢、歷史紀錄、改善建議) 需微調訓練機器人
    

### Acknowledgement

Reference codes:

mediapipe detect:https://github.com/alireza787b/Python-Gaze-Face-Tracker.git

3d landmark detect:https://github.com/JimWest/MeFaMo.git





