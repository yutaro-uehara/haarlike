import numpy as np  
import cv2  
import time  

# 体全体のカスケードファイル  
fullbody_detector = cv2.CascadeClassifier("./haarcascade_fullbody.xml")  
# サンプル画像  
cap = cv2.VideoCapture(0)  

#trackerの作成
tracker = cv2.TrackerMedianFlow_create()

#haarによって検出されたBoundingBoxの格納先
start_rect = (0, 0, 0, 0)


# Shi-Tomasiのコーナー検出パラメータ  
feature_params = dict( maxCorners = 100,  
                       qualityLevel = 0.3,  
                       minDistance = 7,  
                       blockSize = 7 )  

# Lucas-Kanade法のパラメータ  
lk_params = dict( winSize  = (15,15),  
                  maxLevel = 2,  
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))  

# ランダムに色を100個生成（値0～255の範囲で100行3列のランダムなndarrayを生成）  
color = np.random.randint(0, 255, (100, 3))  

# 最初のフレームの処理  
end_flag, frame = cap.read()
start_frame = frame

# グレースケール変換  
gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
# 追跡に向いた特徴  
feature_prev = cv2.goodFeaturesToTrack(gray_prev, mask = None, **feature_params)  
# 元の配列と同じ形にして0を代入  
mask = np.zeros_like(frame)  

while(end_flag):  
    # グレースケールに変換  
    gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

    start = time.time()    
    # 全身の人を検出   
    # minSize:物体が取り得る最小サイズ。これよりも小さい物体は無視される  
    # minNeighbors:物体候補となる矩形は，最低でもこの数だけの近傍矩形を含む  
    body = fullbody_detector.detectMultiScale(gray_next,scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))  
    end = time.time()    
    # 検出時間を表示    
#    print("{} : {:4.1f}ms".format("detectTime", (end - start) * 1000))  

    # オプティカルフロー検出  
    # オプティカルフローとは物体やカメラの移動によって生じる隣接フレーム間の物体の動きの見え方のパターン  
    feature_next, status, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_next, feature_prev, None, **lk_params)  
    # オプティカルフローを検出した特徴点を選別（0：検出せず、1：検出した）  
    good_prev = feature_prev[status == 1]  
    good_next = feature_next[status == 1]  

    # オプティカルフローを描画  
    for i, (next_point, prev_point) in enumerate(zip(good_next, good_prev)):  
        prev_x, prev_y = prev_point.ravel()  
        next_x, next_y = next_point.ravel()  
        mask = cv2.line(mask, (next_x, next_y), (prev_x, prev_y), color[i].tolist(), 2)  
        frame = cv2.circle(frame, (next_x, next_y), 5, color[i].tolist(), -1)  
    img = cv2.add(frame, mask)  

    # 人検出した数表示のため変数初期化  
    human_cnt = 0  
    # 人検出した部分を長方形で囲う  
    for (x, y, w, h) in body:  
        start_rect = (x, y, w, h)
        cv2.rectangle(img, (x, y),(x+w, y+h),(0,255,0),2)  
        # 人検出した数を加算  
        human_cnt += 1
        cv2.putText(img, "Human Cnt:{}".format(int(human_cnt)),(10,550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.imshow('human_view', img)
        break

    # 人検出した数を表示  
    cv2.putText(img, "Human Cnt:{}".format(int(human_cnt)),(10,550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)  
    # ウィンドウに表示  
    cv2.imshow('human_view', img)  

   # ESCキー  
    k = cv2.waitKey(1)  
    if k == 27:  
        break  

    # 次のフレーム、ポイントの準備  
    gray_prev = gray_next.copy()  
    feature_prev = good_next.reshape(-1, 1, 2)  
    end_flag, frame = cap.read()
    start_frame = frame

aa

#トラッキング初期位置の設定
# x: 100, y: 200, width: 30, height: 30
tracker.init(start_frame, start_rect)
while(1):
  frame = cap.read()
  located, bounds = tracker.update(frame)
  if located:
    x = bounds[0]
    y = bounds[1]
    w = bounds[2]
    h = bounds[3]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
  #cv2.imwrite(os.path.join(dst_dir, os.path.basename(files[i])), frame)
  cv2.imshow('human_view', img) 

# 終了処理  
cv2.destroyAllWindows()  
cap.release()  