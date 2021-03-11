# 필요한 모든 라이브러리를 가져옵니다.
import cv2
import numpy as np
import pafy
# 연령 및 성별을 예측할 동영상의 url
# Youtube 동영상 URL 을 가져 와서 webm / mp4 형식으로 동영상의 최상의 해상도를 포함하는 '재생'개체를 만듭니다 .
#url = 'https://www.youtube.com/watch?v=c07IsbSNqfI&feature=youtu.be'
url = 'https://www.youtube.com/watch?v=iTgQWl8J4HQ'
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="mp4")
# 카메라로 라이브 스트림을 캡처해야합니다. OpenCV 는 이에 대한 매우 간단한 인터페이스를 제공합니다.
# 카메라에서 비디오를 캡처하여 회색조 비디오로 변환하여 표시 할 수 있습니다.
# 인수는 장치 인덱스 또는 비디오 파일의 이름 입니다. 장치 인덱스는 카메라를 지정하는 번호입니다.
# 여기에서는 유튜브 url 입니다.

cap = cv2.VideoCapture(play.url)
# 여기서 3 은 너비의 propertyId 이고 4 는 높이입니다.
cap.set(3, 480) # 프레임 너비 설정
cap.set(4, 640) # 프레임 높이 설정
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)','(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']
# 연령 및 성별 감지기의 caffemodel 및 prototxt 를로드하는 함수를 정의
# 기본적으로 감지를 수행 할 사전 훈련 된 CNN 모델입니다
def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('C:/korAI/age_gender_estimation-master/models/deploy_age.prototxt','C:/korAI/age_gender_estimation-master/models/age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('C:/korAI/age_gender_estimation-master/models/deploy_gender.prototxt','C:/korAI/age_gender_estimation-master/models/gender_net.caffemodel')
    return(age_net, gender_net)
# 얼굴 감지, 연령 감지 및 성별 감지를 수행할 video_detector (age_net, gender_net) 함수
def video_detector(age_net, gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        # VideoCapture ()에서 생성된 캡 개체를 읽습니다.
        # cap.read () 는 부울 (True / False)을 반환합니다. 프레임이 올바르게 읽 히면 ret= True 가됩니다.
        ret, image = cap.read()
        # 영상이 잘못 읽히면 또는 끝이면 종료
        if not ret:
            break
        # 괄호 안 따옴표 안 경로에서 검출에 필요한 xml 파일을 불러옴.
        # 얼굴 감지를 위해 미리 빌드 된 모델을로드합니다.
        face_cascade = cv2.CascadeClassifier('C:/korAI/age_gender_estimation-master/models/haarcascade_frontalface_alt.xml')
        # face_cascade = cv2.CascadeClassifier(
        # 'C:/koreaAI/0104/age_gender_estimation-master/models/haarcascade_frontalface_default.xml')
        # OpenCV 얼굴 감지기가 회색 이미지를 예상하므로 이미지를 회색 이미지로 변환합니다
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 필요한 것을 정확히 감지하는 detectMultiScale ()
        # 물체를 감지하는 일반적인 함수
        # 얼굴을 찾으면 "Rect (x, y, w, h)"형식으로 해당 얼굴의 위치 목록을 반환
        # 이미지 : 첫 번째 입력은 Gray 이미지 입니다.  
        # scaleFactor : 이 기능은 잘못된 크기 인식을 보정합니다.
        # minNeighbors : 현재 물체 근처에서 몇 개의 물체가 발견되는지 정의
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if(len(faces) > 0):
            print("Found {} faces".format(str(len(faces))))
            # 얼굴 목록을 반복하고 동영상의 사람 얼굴에 직사각형을 그립니다.
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
            # Get Face
            face_img = image[y:y+h, h:h+w].copy()
            # OpenCV 는 딥 러닝 분류를위한 이미지 전처리를 용이하게하는 기능인 blobFromImage
            # blobFromImage 는 이미지에서 4 차원 blob 을 만듭니다.
            # (image, scalefactor = 1.0, 크기, 평균, swapRB = True)
            # 크기는 Convolutional Neural Network 가 기대하는 공간 크기
            # 평균은 이미지의 모든 채널에서 차감되는 단일 값
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            # 성별을 예측합니다.
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            print("Gender : " + gender)
            # 나이를 예측합니다.
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            print("Age Range: " + age)
            # openCV 의 putText () 모듈을 사용하여 출력 프레임에 텍스트를 넣어야합니다.
            overlay_text = "%s %s" % (gender, age)
            cv2.putText(image, overlay_text, (x, y), font,1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('frame', image)
            # 사용자가 q 키를 누를 때 까지 반복
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
if __name__ == "__main__":
    age_net, gender_net = load_caffe_models()
    video_detector(age_net, gender_net)