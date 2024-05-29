import ultralytics, yaml, torch, metrics
from ultralytics import YOLO

if __name__ == '__main__' :

    # 커스텀 데이터에 맞는 YAML 파일 만들기 
    data = {'train':"E:/img_MNV/train",
            'val':"E:/img_MNV/valid",
            'test': "E:/img_MNV/test",
            'names':['Person','Trolly','Forklift(H)','Forklift(D)', 'Forklilft(V)'],
            'nc' : 5}

    # 지정된 yaml 파일에 작성하기
    with open('mnvision_sample.yaml', 'w') as f:
        yaml.dump(data, f)

    # 작성된 yaml 파일 확인하기
    # with open('mnvision_sample.yaml', 'r') as f:
    #     yolo_yaml = yaml.safe_load(f)
    #     display(yolo_yaml)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(device)

    # YOLO 모델 로딩
    model = YOLO('yolov8n.pt').to(device)  # MS COCO dataset 사전학습된 모델 로딩함 

    # 커스텀 데이터로 학습하기 (반드시 yaml 지정 필요)
    model.train(
        data = 'mnvision_sample.yaml', 
        epochs = 10000, 
        patience = 50, 
        batch = 32, 
        imgsz = 720,
        pretrained = False,
        plots = True,
        degrees = 270,
        translate = 0.12431,
        scale = 0.07643,
        shear = 0.0,
        device = 0,
        optimizer = 'auto',
        val = True,
        verbose = True
        )

    # 검증하기 
    model.val()

    # 검증 결과 출력 
    # print(metrics.box.map)    # mAP (mean Average Precision)
    # print(metrics.box.map50)  # mAP@0.5
    # print(metrics.box.map75)  # mAP@0.75


### model.train 관련 설명 
### https://docs.ultralytics.com/modes/train/#train-settings