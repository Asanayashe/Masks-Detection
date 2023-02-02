import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
from PIL import Image
import torchvision.transforms as T


if __name__ == "__main__":
    transforms = T.Compose([T.ToTensor()])

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)
    model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
    model.eval()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        img = Image.fromarray(frame)
        img = transforms(img),
        pred = model(img)
        for i in range(len((pred[0]['labels']))):
            boxes = list(map(int, pred[0]['boxes'][i]))
            score = pred[0]['scores'][i].detach().numpy().round(2)
            if score > 0.8:
                if pred[0]['labels'][i] == 1:
                    frame = cv2.putText(frame, f"with mask {format(score, '.2f')}", (boxes[0], boxes[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    frame = cv2.rectangle(frame, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (255, 0, 0), 2)
                elif pred[0]['labels'][i] == 2:
                    frame = cv2.putText(frame, f"without mask {format(score, '.2f')}", (boxes[0], boxes[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    frame = cv2.rectangle(frame, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0, 255, 0), 2)
                elif pred[0]['labels'][i] == 3:
                    frame = cv2.putText(frame, f"mask weared incorrect {format(score, '.2f')}", (boxes[0], boxes[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    frame = cv2.rectangle(frame, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0, 0, 255), 2)

        cv2.imshow('Masks Detection', frame)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()